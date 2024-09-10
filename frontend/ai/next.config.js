/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: '20.6.129.131',
        port: '',
        pathname: '/Keyframes/**',
      },
    ],
}
}

module.exports = nextConfig
