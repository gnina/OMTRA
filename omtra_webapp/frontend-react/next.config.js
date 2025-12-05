/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  assetPrefix: '/omtra',
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  // Proxy API requests to avoid CORS and connection issues
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_URL || 'http://api:8000'}/:path*`,
      },
      {
        source: '/omtra/api/:path*',
        destination: `${process.env.API_URL || 'http://api:8000'}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;

