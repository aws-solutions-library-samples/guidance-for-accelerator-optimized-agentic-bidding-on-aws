// AuctionTheater.jsx — a guided, stepped walkthrough of one impression moving
// through the ARTF containers.
//
// The walkthrough is TWO PASSES over the same bid request. Pass 1 auctions it
// with the ARTF extension point asked to propose nothing, so no container is
// consulted. Pass 2 auctions it with the containers mutating it first. Each pass
// opens with a banner, and the recap puts the two outcomes side by side.
//
// The beat sequence is DERIVED from the containers that actually mutated the
// request, so a scenario with three applicable intents produces fewer steps than
// one with six, with no code change.
//
// Everything rendered is a function of (beats, index). Moving backwards produces
// exactly the state that moving forwards to the same index produces, so there is
// no reset-and-replay step and no possibility of residue.

import { useMemo, useEffect, useRef, useState, useCallback } from "react";
import { useTheaterRun, RUN_IDLE, RUN_SUBMITTING, RUN_READY, RUN_FAILED } from "../hooks/useTheaterRun.js";
import { useBeatStepper } from "../hooks/useBeatStepper.js";
import { useRunSummary } from "../hooks/useRunSummary.js";
import {
  visibleValues,
  cardStateFor,
  sawKindInPass,
  BEAT_BIDS,
  BEAT_RECAP,
  BEAT_PASS,
  BEAT_BASELINE,
  PASS_BASELINE,
} from "../utils/theaterBeats.js";
import { factualCaption, mutationNarration } from "../utils/theaterCaptions.js";
import { buildOfferViewModel } from "../utils/offerPresentationService.js";
import { capturedBidResponse } from "../utils/bidResponseFixture.js";
import { deriveSellSideKpis } from "../utils/sellSideKpis.js";
import { compareOutcomes, baselineFacts } from "../utils/outcomeComparison.js";
import {
  TheaterSceneRibbon,
  TheaterRequestCard,
  TheaterSeamArrow,
  TheaterControls,
  CARD_VIEW,
} from "./TheaterPanels.jsx";
import { TheaterMutationStage } from "./TheaterMutationStage.jsx";
import { TheaterPassBanner } from "./TheaterPassBanner.jsx";
import { TheaterRunSummary } from "./TheaterRunSummary.jsx";
import { OffersPanel } from "./OffersPanel.jsx";
import { SellSideDecisionsPanel } from "./SellSideDecisionsPanel.jsx";
import RawPanel from "./RawPanel.jsx";

// `factualCaption` lives in utils/theaterCaptions.js so bedrockCaptionClient can
// degrade to it without a circular import. Re-exported here because it was
// originally defined in this module and is imported from here by tests.
export { factualCaption };

/**
 * The stepped walkthrough for ONE scenario, chosen before it mounts.
 *
 * It used to own a "Choose a scenario" page of its own, because it was a
 * full-screen surface reached from the top nav with no scenario in hand. It is now
 * embedded in the main area and opened from a scenario card, so the scenario and
 * its tuner values arrive as props and the run starts on mount. The chooser is
 * gone rather than hidden: a second place to pick a scenario is a second place for
 * the two to disagree about which tuner values were used.
 */
export default function AuctionTheater({ scenario, params, onExit }) {
  const run = useTheaterRun();
  const stepper = useBeatStepper({ beats: run.beats });

  // Starts the run for whichever scenario is mounted, and re-runs if the parent
  // swaps it. `params` is deliberately NOT a dependency: it is a fresh object on
  // every parent render, so depending on it would resubmit in a loop. The values
  // are read at submit time, which is the moment the button was pressed.
  const startRef = useRef(run.start);
  startRef.current = run.start;
  const paramsRef = useRef(params);
  paramsRef.current = params;
  useEffect(() => {
    if (!scenario) return;
    void startRef.current(scenario, paramsRef.current ?? {});
  }, [scenario?.id]);

  const { beats } = run;
  const { index, currentBeat } = stepper;

  const visible = useMemo(() => visibleValues(beats, index), [beats, index]);
  const landing = useMemo(() => currentBeat?.values ?? [], [currentBeat]);
  const cardState = useMemo(() => cardStateFor(beats, index), [beats, index]);

  const sawOrigin = !!beats && index >= 0;
  const atRecap = currentBeat?.kind === BEAT_RECAP;
  const atPassBanner = currentBeat?.kind === BEAT_PASS;
  const atBaseline = currentBeat?.kind === BEAT_BASELINE;

  // Which of the two auctions the screen is about. Pass 1 is the baseline (no
  // container consulted); pass 2 is the one with ARTF. Before the beats exist
  // there is no pass, and the columns show nothing either way.
  const currentPass = currentBeat?.pass ?? null;

  // The bids are on screen from the pass's BIDS beat onward — including at the
  // beat that closes the pass. Derived from having REACHED the beat rather than
  // from standing on it, because the offers must not disappear when the reader
  // steps past them. Scoped to the pass: reaching pass 1's bids reveals pass 1's
  // offers and says nothing about pass 2's.
  const sawBids = useMemo(
    () => currentPass != null && sawKindInPass(beats, index, BEAT_BIDS, currentPass),
    [beats, index, currentPass],
  );

  // Two bid responses, one per pass.
  //
  // With ARTF: derives entirely from a bid response. Until the Prebid stack is
  // deployed that response is the captured fixture, and the notice the panel shows
  // is chosen from the fixture marker rather than from a prop — so a fixture
  // cannot present as a real auction (FR-31).
  //
  // Without ARTF: no fixture. There is no captured baseline, and standing one in
  // would put an invented number into a comparison whose point is the number. A
  // missing baseline is shown as missing (TP-7).
  const artfResponse = run.bidResponse ?? capturedBidResponse;
  const artfViewModel = useMemo(() => buildOfferViewModel(artfResponse), [artfResponse]);
  const baselineViewModel = useMemo(
    () => (run.baselineResponse ? buildOfferViewModel(run.baselineResponse) : null),
    [run.baselineResponse],
  );

  // What the offers column shows for the pass on screen.
  const offerViewModel = currentPass === PASS_BASELINE ? baselineViewModel : artfViewModel;
  const offerFault = currentPass === PASS_BASELINE ? run.baselineFault : run.auctionFault;
  // The winner is revealed at the beat that closes each pass: `baseline` for pass
  // 1, `recap` for pass 2. The bids beat never leaks it.
  const winnerRevealed = currentPass === PASS_BASELINE ? atBaseline : atRecap;

  // The narration for the step on screen: the container, its intent, and its real
  // values. Derived, never generated — the generated prose is the run summary, and
  // it arrives once, at the end.
  const narration = useMemo(
    () => mutationNarration(currentBeat, run.context),
    [currentBeat, run.context],
  );

  // Both auctions have settled once each has either produced a response or
  // reported why it could not. Waiting for both matters: the summary names the
  // winner and the baseline, so asking before either resolves would describe an
  // auction whose outcome was not yet read.
  const artfSettled = run.bidResponse != null || run.auctionFault != null;
  const baselineSettled = run.baselineResponse != null || run.baselineFault != null;
  const auctionSettled = artfSettled && baselineSettled;

  const baseline = useMemo(
    () => baselineFacts({ baselineVM: baselineViewModel, baselineFault: run.baselineFault }),
    [baselineViewModel, run.baselineFault],
  );

  const summary = useRunSummary({
    beats,
    context: run.context,
    viewModel: artfViewModel,
    baseline,
    ready: run.status === RUN_READY && auctionSettled,
  });

  // The sell-side result for the deal that transacted, with ARTF. Withheld until
  // the recap, for the same reason the offers are: the auction resolves against
  // the enriched request, so showing its result first would put the consequence
  // on screen before its cause.
  const sellSideKpis = useMemo(
    () => (atRecap ? deriveSellSideKpis({ values: visible, context: run.context, viewModel: artfViewModel }) : null),
    [atRecap, visible, run.context, artfViewModel],
  );

  // The two auctions side by side. Recap only: it compares two settled outcomes,
  // and before the recap the with-ARTF outcome has not been revealed.
  const comparison = useMemo(
    () => (atRecap
      ? compareOutcomes({
        baselineVM: baselineViewModel,
        baselineResponse: run.baselineResponse,
        baselineFault: run.baselineFault,
        artfVM: artfViewModel,
        artfResponse: run.bidResponse,
        values: visible,
        context: run.context,
      })
      : null),
    [atRecap, baselineViewModel, run.baselineResponse, run.baselineFault, artfViewModel, run.bidResponse, visible, run.context],
  );

  /* ------------------------------------------------------ card view + summary */

  const [cardView, setCardView] = useState(CARD_VIEW.VISUAL);
  const [summaryOpen, setSummaryOpen] = useState(false);
  // Which run has already auto-opened its summary, compared by identity. Without
  // it, dismissing the summary and stepping back to the recap would reopen it —
  // the reader's dismissal has to stick.
  const autoShownRef = useRef(null);

  useEffect(() => {
    if (!atRecap || !beats) return;
    if (autoShownRef.current === beats) return;
    autoShownRef.current = beats;
    setSummaryOpen(true);
    setCardView(CARD_VIEW.INFO);
  }, [atRecap, beats]);

  const closeSummary = useCallback(() => {
    setSummaryOpen(false);
    // The info view is the summary. Leaving the toggle on `info` with the surface
    // dismissed would show a pressed control over a body explaining that the
    // surface is dismissed, so it returns to the visual view.
    setCardView((v) => (v === CARD_VIEW.INFO ? CARD_VIEW.VISUAL : v));
  }, []);

  const changeCardView = useCallback((next) => {
    setCardView(next);
    if (next === CARD_VIEW.INFO) setSummaryOpen(true);
    else setSummaryOpen(false);
  }, []);

  const stepLabel = stepper.isIndeterminate
    ? "Preparing"
    : `Step ${index + 1} of ${stepper.total}`;

  // The same merged JSON the default scenario view renders, through the same
  // component. A response-side scenario carries `bid_response` in its payload, so
  // bid shading is shown against the response without a second code path.
  const codeSlot = (
    <RawPanel result={run.result} payload={run.payload} section="request" />
  );

  const infoSlot = (
    <div className="th-card-info">
      <p className="th-card-info-text">{summary.text || "The run has not finished yet."}</p>
      {summary.text ? (
        <button
          type="button"
          className="th-btn th-btn-ghost th-card-info-reopen"
          onClick={() => setSummaryOpen(true)}
          data-testid="card-info-reopen"
        >
          Show over the stage
        </button>
      ) : null}
    </div>
  );

  return (
    <div className="th-stage th-stage-embedded">
      {/* The scenario name and the Close control used to sit in their own bar
          above the ribbon. They are in the ribbon now: one row instead of two,
          which is vertical space the columns get back on a laptop screen. */}
      <TheaterSceneRibbon
        context={run.context}
        revealed={sawOrigin && run.status === RUN_READY}
        stepLabel={stepLabel}
        scenarioName={scenario?.name ?? "No scenario"}
        onExit={onExit}
      />

      {/* RUN_IDLE is now only the instant between mount and the start effect
          firing. It is not a chooser any more — there is nothing to choose. */}
      {run.status === RUN_IDLE || run.status === RUN_SUBMITTING ? (
        <div className="th-chooser">
          <h2 className="th-chooser-title">Submitting to the orchestrator</h2>
          <p className="th-chooser-sub">
            The number of steps is not known yet. It depends on how many mutations
            the containers return.
          </p>
        </div>
      ) : null}

      {run.status === RUN_FAILED ? (
        <div className="th-chooser">
          <h2 className="th-chooser-title">The request was not processed</h2>
          <p className="th-chooser-sub th-error">
            {run.error?.message ?? "The orchestrator could not be reached."}
          </p>
          <p className="th-chooser-sub">
            There is no walkthrough to show, because nothing was processed.
          </p>
          <div className="th-chooser-actions">
            {/* Retry re-submits THIS scenario. There is no "choose another" any
                more; closing returns to the picker, which is where scenarios are
                chosen. */}
            <button
              type="button"
              className="th-btn th-btn-primary"
              onClick={() => run.start(scenario, params ?? {})}
              data-testid="theater-retry"
            >
              Try again
            </button>
            <button type="button" className="th-btn th-btn-ghost" onClick={onExit}>
              Close
            </button>
          </div>
        </div>
      ) : null}

      {run.status === RUN_READY ? (
        <>
          <div className="th-body">
            <div className="th-columns">
              <SellSideDecisionsPanel
                values={visible}
                kpis={sellSideKpis}
                comparison={comparison}
                revealed={visible.length > 0}
              />
              {/* Centre column — ARTF mutations on the request (FR-24), in
                  whichever of the three views is selected. */}
              <div className="th-col th-col-centre">
                <TheaterRequestCard
                  context={run.context}
                  visible={visible}
                  landingValues={landing}
                  cardState={cardState}
                  contributors={atRecap ? currentBeat.contributors : null}
                  view={cardView}
                  onViewChange={changeCardView}
                  codeSlot={codeSlot}
                  infoSlot={infoSlot}
                />
              </div>
              {/*
                Right column: the offers returned under one seat, with each
                candidate's outcome and, where it made no offer, its reason.
              */}
              <OffersPanel
                viewModel={offerViewModel}
                revealed={sawBids}
                winnerRevealed={winnerRevealed}
                auctionFault={offerFault}
                baseline={currentPass === PASS_BASELINE}
              />
            </div>

            <TheaterSeamArrow movement={currentBeat?.movement} active />

            {/* The pass banner, over the whole stage, while the stepper is on a
                pass beat. It handles its own fade-out, so it is always mounted
                and told which beat (if any) it is announcing. */}
            <TheaterPassBanner beat={atPassBanner ? currentBeat : null} />

            {/* Over the centre of the stage, above the columns. Suppressed while
                the summary is showing so two surfaces never contend for the same
                middle of the screen.

                Also suppressed on the recap beat. This card names which container
                changed the request under which intent; the recap has no container
                and no intent, so it rendered a bare step label over a headline
                count the summary already states — and, being the last beat, it
                could not be advanced past, so dismissing the summary left it
                covering the request card with no way to clear it.

                And suppressed on a pass beat, whose surface is the banner. */}
            {!summaryOpen && !atRecap && !atPassBanner ? (
              <TheaterMutationStage narration={narration} stepLabel={stepLabel} />
            ) : null}

            <TheaterRunSummary
              text={summary.text}
              open={summaryOpen}
              onClose={closeSummary}
              subtitle={artfViewModel?.noticeText}
              comparison={comparison}
            />
          </div>

          {/* A footer in flow, not a floating pill. It used to overlap the data it
              was controlling. */}
          <TheaterControls
            index={index}
            total={stepper.total}
            isIndeterminate={stepper.isIndeterminate}
            isPlaying={stepper.isPlaying}
            canGoBack={stepper.canGoBack}
            canGoNext={stepper.canGoNext}
            onNext={stepper.next}
            onBack={stepper.back}
            onRestart={stepper.restart}
            onTogglePlay={stepper.togglePlay}
            onExit={onExit}
          />
        </>
      ) : null}
    </div>
  );
}
