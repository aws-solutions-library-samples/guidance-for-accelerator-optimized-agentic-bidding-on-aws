import { SCENARIOS } from '../components/ScenarioCard.jsx';

/**
 * DemoSequenceOrchestrator — Builds and manages the demo walkthrough sequence.
 *
 * Knows about SCENARIOS, submission logic, and focus-step ordering.
 * Cycles through all 8 scenarios, submitting each via the provided submitFn,
 * building animation sequences from results, and feeding them to the engine.
 *
 * This is a plain class (not a React component) — instantiated by the hook/component.
 */
class DemoSequenceOrchestrator {
  /**
   * @param {import('./AnimationEngine.js').default} engine - AnimationEngine instance
   * @param {(scenario: object) => Promise<object>} submitFn - Function that submits a scenario and returns the result
   * @param {object} [options]
   * @param {number} [options.maxConsecutiveFailures=3] - Stop after this many consecutive failures
   * @param {number} [options.interScenarioDelayMs=1000] - Delay between scenarios
   */
  constructor(engine, submitFn, options = {}) {
    this._engine = engine;
    this._submitFn = submitFn;
    this._options = {
      maxConsecutiveFailures: options.maxConsecutiveFailures ?? 3,
      interScenarioDelayMs: options.interScenarioDelayMs ?? 1000,
    };

    this._running = false;
    this._stopped = false;
    this._currentScenarioIndex = 0;
    this._consecutiveFailures = 0;
    this._delayTimeout = null;
  }

  /**
   * Whether the orchestrator is currently running.
   * @returns {boolean}
   */
  get isRunning() {
    return this._running;
  }

  /**
   * Current scenario index (0-based).
   * @returns {number}
   */
  get currentScenarioIndex() {
    return this._currentScenarioIndex;
  }

  /**
   * Start the demo loop. Cycles through all scenarios, looping indefinitely
   * until stop() is called.
   * @returns {Promise<void>}
   */
  async start() {
    if (this._running) return;

    this._running = true;
    this._stopped = false;
    this._consecutiveFailures = 0;
    this._currentScenarioIndex = 0;

    try {
      while (!this._stopped) {
        const scenario = SCENARIOS[this._currentScenarioIndex];

        try {
          // Submit the scenario via the provided function
          const result = await this._submitFn(scenario);

          if (this._stopped) break;

          // Reset consecutive failure counter on success
          this._consecutiveFailures = 0;

          // Build animation sequence from the result
          const sequence = this.buildSequenceForResult(result, scenario);

          // Feed to engine
          await this._engine.execute(sequence);

          if (this._stopped) break;
        } catch (err) {
          if (this._stopped) break;

          this._consecutiveFailures++;
          console.warn(
            `[DemoSequenceOrchestrator] Scenario "${scenario.name}" failed (${this._consecutiveFailures}/${this._options.maxConsecutiveFailures}):`,
            err.message || err
          );

          // Show error annotation via engine (brief error sequence)
          await this._showErrorAnnotation(scenario, err);

          if (this._stopped) break;

          // Stop after too many consecutive failures
          if (this._consecutiveFailures >= this._options.maxConsecutiveFailures) {
            console.error(
              '[DemoSequenceOrchestrator] Stopping — too many consecutive failures.'
            );
            break;
          }
        }

        if (this._stopped) break;

        // Advance to next scenario, looping back to 0
        this._currentScenarioIndex =
          (this._currentScenarioIndex + 1) % SCENARIOS.length;

        // Brief pause between scenarios
        await this._delay(this._options.interScenarioDelayMs);
      }
    } finally {
      this._running = false;
    }
  }

  /**
   * Stop the demo loop and restore UI.
   * @returns {Promise<void>}
   */
  async stop() {
    this._stopped = true;

    // Clear any pending inter-scenario delay
    if (this._delayTimeout) {
      clearTimeout(this._delayTimeout);
      this._delayTimeout = null;
    }

    // Stop the engine (reverses transforms, clears annotations)
    await this._engine.stop();

    this._running = false;
  }

  /**
   * Build the Animation_Sequence for a single scenario result.
   *
   * Produces steps in order: timeline → request-json → mutation-line-{i}
   *
   * @param {object} result - Normalized orchestrator API response
   * @param {object} scenario - SCENARIOS entry
   * @returns {Array<{ focusArea: string, zoomLevel: number, annotation: string|null, dwellMs: number, order: number }>}
   */
  buildSequenceForResult(result, scenario) {
    const steps = [];
    let order = 0;

    // Gather all mutations from stops
    const allMutations = (result?.stops || []).flatMap(
      (stop) => stop.mutations || []
    );
    // Count only stops with latency (actual agent processing, not pass-through SSP/DSP)
    const agentStops = (result?.stops || []).filter(s => s.latency);
    const totalLatencyMs = result?.totalLatencyMs || 0;

    // Step 1: Focus on timeline
    steps.push({
      focusArea: 'timeline',
      zoomLevel: 1.3,
      annotation: `${scenario.name} — ${agentStops.length} agents processed in ${totalLatencyMs}ms`,
      dwellMs: 3000,
      order: order++,
    });

    // Step 2: Focus on request JSON
    steps.push({
      focusArea: 'request-json',
      zoomLevel: 1.2,
      annotation: `Request with ${allMutations.length} mutations applied`,
      dwellMs: 2500,
      order: order++,
    });

    // Step 3+: Each mutation highlight line
    for (let i = 0; i < allMutations.length; i++) {
      const mutation = allMutations[i];
      const agent = mutation.agent || mutation.source || 'Agent';
      const intent = mutation.intent || 'mutation';

      steps.push({
        focusArea: `mutation-line-${i + 1}`,
        zoomLevel: 1.4,
        annotation: `${agent}: ${intent}`,
        dwellMs: 2000,
        order: order++,
      });
    }

    return steps;
  }

  /**
   * Show a brief error annotation when a scenario fails.
   * @param {object} scenario
   * @param {Error} err
   * @returns {Promise<void>}
   * @private
   */
  async _showErrorAnnotation(scenario, err) {
    const errorSequence = [
      {
        focusArea: 'timeline',
        zoomLevel: 1.0,
        annotation: `Failed: ${scenario.name} — ${err.message || 'API error'}`,
        dwellMs: 2000,
        order: 0,
      },
    ];

    try {
      await this._engine.execute(errorSequence);
    } catch {
      // Swallow — we're already handling an error
    }
  }

  /**
   * Delay for the specified duration. Can be cancelled by stop().
   * @param {number} ms
   * @returns {Promise<void>}
   * @private
   */
  _delay(ms) {
    return new Promise((resolve) => {
      this._delayTimeout = setTimeout(() => {
        this._delayTimeout = null;
        resolve();
      }, ms);
    });
  }
}

export default DemoSequenceOrchestrator;
