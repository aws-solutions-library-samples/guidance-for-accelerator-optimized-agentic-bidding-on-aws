import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    // RTB Fabric endpoint — in production, CloudFront routes /fabric/* to RTB Fabric.
    // In local dev, both paths hit the same orchestrator (demonstrating the UI toggle).
    "__ARTF_FABRIC_URL__": JSON.stringify(process.env.ARTF_FABRIC_URL || "/fabric"),
  },
  server: {
    proxy: {
      // Standalone path: /api/* → orchestrator
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // RTB Fabric path: /fabric/* → same orchestrator in local dev
      // In production, CloudFront routes this through RTB Fabric infrastructure
      "/fabric": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/fabric/, ""),
        // Add header to simulate fabric path identification
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("X-RTB-Fabric-Simulated", "local-dev");
          });
        },
      },
    },
  },
});
