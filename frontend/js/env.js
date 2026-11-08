// Environment configuration for SentimentLens frontend
// Automatically switches between local development and production Render backend

const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

// When you deploy your backend to Render, put your Render URL here:
// Example: "https://sentimentlens-api.onrender.com"
const PRODUCTION_RENDER_URL = "https://dl-bg2h.onrender.com";

const ENV = {
    // Uses localhost:8001 when testing locally; uses Render URL on Vercel
    API_BASE: isLocal
        ? "http://127.0.0.1:8001"
        : (localStorage.getItem("CUSTOM_API_BASE") || PRODUCTION_RENDER_URL),

    DEBUG: isLocal,

    // UI Settings
    MAX_TEXT_LENGTH: 5000,
    MIN_TEXT_LENGTH: 10,

    // Request timeout (ms) - 60 seconds accommodates Render free-tier cold starts
    REQUEST_TIMEOUT: 60000,

    // Health check timeout (ms)
    HEALTH_CHECK_TIMEOUT: 10000,

    // Periodic health check interval (ms) - 2 minutes (120,000 ms) prevents repeated signals
    HEALTH_CHECK_INTERVAL: 120000
};
