import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development (npm run dev) the API runs separately on port 8000.
// In production the Python server serves the built files and the API together.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", emptyOutDir: true },
});
