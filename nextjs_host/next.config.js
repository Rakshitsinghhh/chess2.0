/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ["onnxruntime-node"]
  }
};

module.exports = nextConfig;

