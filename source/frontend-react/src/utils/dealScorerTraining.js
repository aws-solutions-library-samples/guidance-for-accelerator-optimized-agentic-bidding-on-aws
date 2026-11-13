// How the Deal Scorer this deployment serves was trained.
//
// deploy.sh Step 2 trains it on sample interactions generated from the deal book
// and the cohort spec (source/training/ncf_sample/), gates it against the scenario
// samples, and labels the build ncf-neumf-sample-<fingerprint>; the container
// refuses to start on Triton without such a label. The disclosure is shown wherever the
// Deal Scorer's decisions are: its theater beats and the scenario cards it scores.

export const DEAL_SCORER_TRAINING_DISCLOSURE = "Deal Scorer trained on sample data";

/** model_version prefix of a sample-trained build (source/shared/ncf_ids.py SAMPLE_VERSION_PREFIX). */
export const SAMPLE_MODEL_VERSION_PREFIX = "ncf-neumf-sample-";

/** The Deal Scorer's stop id (intentMapping.js) and its scenario model key. */
export const DEAL_SCORER_STOP_ID = "ncf";
export const DEAL_SCORER_MODEL_KEY = "ncf_deal_manager";

/**
 * Whether a container beat carries the disclosure.
 *
 * True for the Deal Scorer stop unless the beat names a model_version without the
 * sample prefix. The normalized stop carries no model_version today, so on the
 * Deal Scorer stop a null version is the deployed build.
 */
export function showsDealScorerDisclosure(containerName, modelVersion) {
  if (containerName !== DEAL_SCORER_STOP_ID) return false;
  if (typeof modelVersion !== "string" || modelVersion === "") return true;
  return modelVersion.startsWith(SAMPLE_MODEL_VERSION_PREFIX);
}

/** Whether a scenario card's models include the Deal Scorer. */
export function scenarioUsesDealScorer(scenario) {
  return Array.isArray(scenario?.models)
    && scenario.models.some((m) => m?.key === DEAL_SCORER_MODEL_KEY);
}
