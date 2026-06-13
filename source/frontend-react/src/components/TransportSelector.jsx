import { useCallback } from "react";

export default function TransportSelector({ transport, setTransport }) {
  const handleSelect = useCallback(
    (value) => {
      setTransport(value);
    },
    [setTransport]
  );

  return (
    <div className="transport-pill" role="radiogroup" aria-label="Transport protocol">
      <button
        className={`transport-pill-option ${transport === "REST" ? "active" : ""}`}
        role="radio"
        aria-checked={transport === "REST"}
        onClick={() => handleSelect("REST")}
      >
        REST
      </button>
      <button
        className={`transport-pill-option ${transport === "MCP" ? "active" : ""}`}
        role="radio"
        aria-checked={transport === "MCP"}
        onClick={() => handleSelect("MCP")}
      >
        MCP
      </button>
    </div>
  );
}
