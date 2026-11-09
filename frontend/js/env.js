// Environment configuration for dl frontend
// Automatically switches between local development and production Render backend

const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

// When you deploy your backend to Render, put your Render URL here:
// Example: "https://your-api.onrender.com"
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

    // Health check timeout (ms) - 45 seconds accommodates Render free-tier cold starts
    HEALTH_CHECK_TIMEOUT: 45000,

    // Keep-alive and periodic health check interval (ms) - 3.5 minutes keeps Render awake
    HEALTH_CHECK_INTERVAL: 210000,
    KEEP_ALIVE_INTERVAL: 210000
};
