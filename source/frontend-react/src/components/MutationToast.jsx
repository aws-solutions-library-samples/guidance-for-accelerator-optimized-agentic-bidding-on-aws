/**
 * MutationToast — overlay showing mutation details on agent nodes.
 * Appears when the animated dot reaches an agent node.
 */
export default function MutationToast({ mutations, visible }) {
  if (!visible || !mutations || mutations.length === 0) return null;

  return (
    <div className="flow-mutation-toast" aria-live="polite">
      {mutations.map((m, i) => (
        <div key={i}>
          <span style={{ color: "var(--green)", fontWeight: 600 }}>{m.intent}</span>
          {" → "}
          <span style={{ color: "var(--aws)" }}>{m.op}</span>
          {m.path && (
            <>
              {" "}
              <span style={{ color: "var(--accent-light)" }}>{m.path}</span>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
