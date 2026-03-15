/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  // Only use assetPrefix in production (when deployed with /omtra path)
  assetPrefix: process.env.NODE_ENV === 'production' ? '/omtra' : undefined,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  // Proxy API requests to avoid CORS and connection issues
  async rewrites() {
    // Use localhost when running dev server outside Docker, otherwise use Docker service name
    // Check if we're in development mode (dev server) vs production (Docker)
    const isDev = process.env.NODE_ENV !== 'production';
    // Hardcode API URL to avoid environment variable issues in Docker
    const apiUrl = isDev ? 'http://localhost:8000' : 'http://api:8000';
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

