/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  // Only use assetPrefix in production (when deployed with /omtra path)
  assetPrefix: process.env.NODE_ENV === 'production' ? '/omtra' : undefined,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    // Optional override when the UI is not at site root (e.g. set to "/omtra/api").
    NEXT_PUBLIC_API_BASE_PATH: process.env.NEXT_PUBLIC_API_BASE_PATH || '',
  },
  // Proxy API requests to avoid CORS and connection issues
  async rewrites() {
    // Prefer API_URL when set (Docker compose sets http://api:8000). For `next dev` on the
    // host without Docker, fall back to localhost on the default API port.
    const apiUrl =
      process.env.API_URL ||
      (process.env.NODE_ENV !== 'production'
        ? 'http://localhost:8000'
        : 'http://api:8000');
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/:path*`,
      },
      {
        source: '/omtra/api/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
  // Prevent browser caching issues after rebuilds
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'no-cache, no-store, must-revalidate',
          },
        ],
      },
    ];
  },
  // Generate unique build ID to force cache invalidation
  generateBuildId: async () => {
    return `build-${Date.now()}`;
  },
};

module.exports = nextConfig;

