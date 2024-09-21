/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: '104.214.176.95',
        port: '',
        pathname: '/Keyframes/**',
      },
    ],
}
}

module.exports = nextConfig
