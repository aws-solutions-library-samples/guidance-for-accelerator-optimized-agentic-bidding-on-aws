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

import { useMemo } from "react";
import { SCENARIOS } from "./ScenarioCard.jsx";
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

export default function AuctionTheater({ onExit }) {
  const run = useTheaterRun();
  const stepper = useBeatStepper({ beats: run.beats });

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
    <div className="th-stage">
      <TheaterSceneRibbon
        context={run.context}
        revealed={sawOrigin && run.status === RUN_READY}
        stepLabel={stepLabel}
      />

      {run.status === RUN_IDLE ? (
        <div className="th-chooser">
          <h2 className="th-chooser-title">Choose a scenario</h2>
          <p className="th-chooser-sub">
            Each scenario is submitted to the running orchestrator. The walkthrough
            has one step for every mutation the containers actually return, so its
            length depends on the scenario.
          </p>
          <div className="th-chooser-list" data-testid="theater-scenario-select">
            {SCENARIOS.map((s) => (
              <button key={s.id} type="button" className="th-scenario"
                onClick={() => run.start(s)}>
                <span className="th-scenario-name">{s.name}</span>
                <span className="th-scenario-desc">{s.desc}</span>
              </button>
            ))}
          </div>
          <button type="button" className="th-btn th-btn-ghost th-chooser-exit"
            onClick={onExit} data-testid="theater-exit">Exit</button>
        </div>
      ) : null}

      {run.status === RUN_SUBMITTING ? (
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
            <button type="button" className="th-btn th-btn-primary" onClick={run.reset}>
              Choose a scenario
            </button>
            <button type="button" className="th-btn th-btn-ghost" onClick={onExit}>
              Exit
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
