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
      <div className="container-wide py-8 sm:py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {/* Brand Section */}
          <div className="col-span-1 sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="h-8 w-8 rounded-md fashion-gradient flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
                StyleAI
              </h3>
            </div>
            <p className="text-muted mb-4 text-sm sm:text-base max-w-md">
              Your AI-powered fashion companion. Get personalized style recommendations based on your skin tone and preferences.
            </p>
            <div className="flex gap-3">
              <a href="#" className="text-muted hover:text-primary-600 transition-colors p-2 rounded-md hover:bg-surface-hover">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-muted hover:text-primary-600 transition-colors p-2 rounded-md hover:bg-surface-hover">
                <Instagram className="h-5 w-5" />
              </a>
              <a href="#" className="text-muted hover:text-primary-600 transition-colors p-2 rounded-md hover:bg-surface-hover">
                <Github className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-foreground mb-3 text-sm sm:text-base">Quick Links</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/analyze" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Analyze Photo
                </Link>
              </li>
              <li>
                <Link href="/wardrobe" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  My Wardrobe
                </Link>
              </li>
              <li>
                <Link href="/recommendations" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Get Recommendations
                </Link>
              </li>
              <li>
                <Link href="/mix-match" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Mix & Match
                </Link>
              </li>
            </ul>
          </div>

          {/* Features */}
          <div>
            <h4 className="font-semibold text-foreground mb-3 text-sm sm:text-base">Features</h4>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Color Analysis
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Smart Wardrobe
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  AI Matching
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Style Guide
                </a>
              </li>
            </ul>
          </div>

          {/* Support */}
          <div>
            <h4 className="font-semibold text-foreground mb-3 text-sm sm:text-base">Support</h4>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Help Center
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Terms of Service
                </a>
              </li>
              <li>
                <a href="#" className="text-muted hover:text-primary-600 transition-colors text-sm">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-6 sm:mt-8 pt-6 sm:pt-8 border-t border-border">
          <p className="text-center text-muted text-xs sm:text-sm flex flex-wrap items-center justify-center gap-1">
            Made with <Heart className="h-4 w-4 text-red-500 fill-red-500" /> by StyleAI Team © 2025
          </p>
        </div>
      </div>
    </footer>
  );
}

