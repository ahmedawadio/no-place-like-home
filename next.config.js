/** @type {import('next').NextConfig} */
const nextConfig = {
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
      
        
        destination:
          process.env.NODE_ENV === 'development'
            ? 'http://127.0.0.1:5328/api/:path*'
            : '/api/index.py', // In production, direct to main API handler
      
            
      },
    ]
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        port: '', // Leave empty if there's no specific port.
        pathname: '/**', // Allows all paths from Unsplash.
      },
      {
        protocol: 'https',
        hostname: 'assets.aceternity.com',
        port: '', // Leave empty if there's no specific port.
        pathname: '/**', // Allows all paths from Unsplash.
      },
    ],
  },
}

module.exports = nextConfig
