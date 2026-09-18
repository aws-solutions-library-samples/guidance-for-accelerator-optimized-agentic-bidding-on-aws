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

import { useMemo, useEffect, useRef } from "react";
import { useTheaterRun, RUN_IDLE, RUN_SUBMITTING, RUN_READY, RUN_FAILED } from "../hooks/useTheaterRun.js";
import { useBeatStepper } from "../hooks/useBeatStepper.js";
import { useBeatCaption } from "../hooks/useBeatCaption.js";
import { visibleValues, cardStateFor, BEAT_RECAP } from "../utils/theaterBeats.js";
import { factualCaption } from "../utils/theaterCaptions.js";
import { buildOfferViewModel } from "../utils/offerPresentationService.js";
import { capturedBidResponse } from "../utils/bidResponseFixture.js";
import {
  TheaterSceneRibbon,
  TheaterRequestCard,
  TheaterCaption,
  TheaterSeamArrow,
  TheaterControls,
} from "./TheaterPanels.jsx";
import { OffersPanel } from "./OffersPanel.jsx";
import { SellSideDecisionsPanel } from "./SellSideDecisionsPanel.jsx";

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

  // The caption is generated where possible and states the beat's real values
  // where not. The hook writes the factual caption synchronously on beat change,
  // so there is never a moment without one, and it discards a generated caption
  // that arrives after the reader has moved on.
  const caption = useBeatCaption({ beat: currentBeat, context: run.context });

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

  const stepLabel = stepper.isIndeterminate
    ? "Preparing"
    : `Step ${index + 1} of ${stepper.total}`;

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
          <div className="th-columns">
            <SellSideDecisionsPanel
              values={visible}
              consequences={run.sellSideConsequences}
              revealed={visible.length > 0}
            />
            {/* Centre column unchanged — ARTF mutations on the request (FR-24). */}
            <div className="th-col th-col-centre">
              <TheaterRequestCard
                context={run.context}
                visible={visible}
                landingValues={landing}
                cardState={cardState}
                contributors={atRecap ? currentBeat.contributors : null}
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

          <TheaterCaption
            text={caption.text}
            stepLabel={stepLabel}
            recap={atRecap ? currentBeat.contributors : null}
          />

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
