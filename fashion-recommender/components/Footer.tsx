import Link from 'next/link'
import { Sparkles, Instagram, Twitter, Facebook, Mail, Heart, Github } from 'lucide-react'

export default function Footer() {
  const currentYear = new Date().getFullYear()

  const footerSections = [
    {
      title: 'Fashion AI',
      links: [
        { href: '/analyze', label: 'Style Analysis' },
        { href: '/wardrobe', label: 'Wardrobe Manager' },
        { href: '/recommendations', label: 'AI Recommendations' },
        { href: '/mix-match', label: 'Mix & Match' },
      ],
    },
    {
      title: 'Features',
      links: [
        { href: '/color-analysis', label: 'Color Analysis' },
        { href: '/body-type', label: 'Body Type Guide' },
        { href: '/trends', label: 'Fashion Trends' },
        { href: '/seasonal', label: 'Seasonal Colors' },
      ],
    },
    {
      title: 'Community',
      links: [
        { href: '/blog', label: 'Fashion Blog' },
        { href: '/stylists', label: 'Expert Stylists' },
        { href: '/inspiration', label: 'Style Inspiration' },
        { href: '/challenges', label: 'Style Challenges' },
      ],
    },
    {
      title: 'Support',
      links: [
        { href: '/help', label: 'Help Center' },
        { href: '/contact', label: 'Contact Us' },
        { href: '/faq', label: 'FAQ' },
        { href: '/feedback', label: 'Send Feedback' },
      ],
    },
  ]

  const socialLinks = [
    { href: 'https://instagram.com', label: 'Instagram', icon: Instagram },
    { href: 'https://twitter.com', label: 'Twitter', icon: Twitter },
    { href: 'https://facebook.com', label: 'Facebook', icon: Facebook },
    { href: 'https://github.com', label: 'GitHub', icon: Github },
    { href: 'mailto:hello@styleai.com', label: 'Email', icon: Mail },
  ]

  return (
    <footer className="border-t border-border bg-surface transition-theme">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 sm:col-span-2">
            <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              StyleAI
            </h3>
            <p className="text-gray-600 mb-4 max-w-md">
              Your AI-powered fashion companion. Get personalized style recommendations based on your skin tone and preferences.
            </p>
            <div className="flex gap-4">
              <a href="#" className="text-gray-400 hover:text-purple-600 transition-colors">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-purple-600 transition-colors">
                <Instagram className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-purple-600 transition-colors">
                <Github className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/analyze" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Analyze Photo
                </Link>
              </li>
              <li>
                <Link href="/wardrobe" className="text-gray-600 hover:text-purple-600 transition-colors">
                  My Wardrobe
                </Link>
              </li>
              <li>
                <Link href="/recommendations" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Get Recommendations
                </Link>
              </li>
              <li>
                <Link href="/mix-match" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Mix & Match
                </Link>
              </li>
            </ul>
          </div>

          {/* About */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">About</h4>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-gray-600 hover:text-purple-600 transition-colors">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Terms of Service
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-600 hover:text-purple-600 transition-colors">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-8 border-t border-gray-200">
          <p className="text-center text-gray-600 text-sm flex flex-wrap items-center justify-center gap-1">
            Made with <Heart className="h-4 w-4 text-red-500 fill-red-500" /> by StyleAI Team © 2025
          </p>
        </div>
      </div>
    </footer>
  );
}

