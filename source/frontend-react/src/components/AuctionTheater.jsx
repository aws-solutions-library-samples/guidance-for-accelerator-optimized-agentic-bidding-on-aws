// AuctionTheater.jsx — a guided, stepped walkthrough of one impression moving
// through the ARTF containers.
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
import { visibleValues, cardStateFor, BEAT_RECAP } from "../utils/theaterBeats.js";
import { factualCaption, mutationNarration } from "../utils/theaterCaptions.js";
import { buildOfferViewModel } from "../utils/offerPresentationService.js";
import { capturedBidResponse } from "../utils/bidResponseFixture.js";
import { deriveSellSideKpis } from "../utils/sellSideKpis.js";
import {
  TheaterSceneRibbon,
  TheaterRequestCard,
  TheaterSeamArrow,
  TheaterControls,
  CARD_VIEW,
} from "./TheaterPanels.jsx";
import { TheaterMutationStage } from "./TheaterMutationStage.jsx";
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

  // The offers column derives entirely from a bid response. Until the Prebid stack
  // is deployed that response is the captured fixture, and the notice the panel
  // shows is chosen from the fixture marker rather than from a prop — so a fixture
  // cannot present as a real auction (FR-31).
  const bidResponse = run.bidResponse ?? capturedBidResponse;
  const offerViewModel = useMemo(() => buildOfferViewModel(bidResponse), [bidResponse]);

  // The narration for the step on screen: the container, its intent, and its real
  // values. Derived, never generated — the generated prose is the run summary, and
  // it arrives once, at the end.
  const narration = useMemo(
    () => mutationNarration(currentBeat, run.context),
    [currentBeat, run.context],
  );

  // The auction has settled once it has either produced a response or reported why
  // it could not. Waiting for it matters: the summary names the winner, so asking
  // before it resolves would describe an auction whose outcome was not yet read.
  const auctionSettled = run.bidResponse != null || run.auctionFault != null;
  const summary = useRunSummary({
    beats,
    context: run.context,
    viewModel: offerViewModel,
    ready: run.status === RUN_READY && auctionSettled,
  });

  // The sell-side result for the deal that transacted. Withheld until the offers
  // are revealed, for the same reason the offers are: the auction resolves against
  // the enriched request, so showing its result first would put the consequence on
  // screen before its cause.
  const sellSideKpis = useMemo(
    () => (atRecap ? deriveSellSideKpis({ values: visible, context: run.context, viewModel: offerViewModel }) : null),
    [atRecap, visible, run.context, offerViewModel],
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
      <div className="th-embedded-bar">
        <span className="th-embedded-scenario" data-testid="theater-scenario-name">
          {scenario?.name ?? "No scenario"}
        </span>
        <button
          type="button"
          className="th-btn th-btn-ghost th-embedded-close"
          onClick={onExit}
          data-testid="theater-exit"
        >
          Close
        </button>
      </div>

      <TheaterSceneRibbon
        context={run.context}
        revealed={sawOrigin && run.status === RUN_READY}
        stepLabel={stepLabel}
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
                revealed={atRecap}
                auctionFault={run.auctionFault}
              />
            </div>

            <TheaterSeamArrow movement={currentBeat?.movement} active />

            {/* Over the centre of the stage, above the columns. Suppressed while
                the summary is showing so two surfaces never contend for the same
                middle of the screen.

                Also suppressed on the recap beat. This card names which container
                changed the request under which intent; the recap has no container
                and no intent, so it rendered a bare step label over a headline
                count the summary already states — and, being the last beat, it
                could not be advanced past, so dismissing the summary left it
                covering the request card with no way to clear it. */}
            {!summaryOpen && !atRecap ? (
              <TheaterMutationStage narration={narration} stepLabel={stepLabel} />
            ) : null}

            <TheaterRunSummary
              text={summary.text}
              open={summaryOpen}
              onClose={closeSummary}
              subtitle={offerViewModel?.noticeText}
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
