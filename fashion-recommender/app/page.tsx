import Link from 'next/link';
import { Camera, Shirt, Sparkles, Target, ArrowRight, CheckCircle2 } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: Camera,
      title: 'Skin Tone Analysis',
      description: 'Upload your photo and let our AI analyze your skin tone and undertone for personalized recommendations.',
    },
    {
      icon: Shirt,
      title: 'Smart Wardrobe',
      description: 'Digitize your wardrobe by uploading photos of your clothing items for intelligent outfit suggestions.',
    },
    {
      icon: Sparkles,
      title: 'AI Recommendations',
      description: 'Get personalized outfit recommendations based on color theory, occasion, season, and your style.',
    },
    {
      icon: Target,
      title: 'Compatibility Checker',
      description: 'Mix and match items to see how well they work together with real-time compatibility scores.',
    },
  ];

  const benefits = [
    'Never worry about color matching again',
    'Discover new outfit combinations from your existing wardrobe',
    'Get outfit suggestions for any occasion',
    'Save time deciding what to wear',
    'Build confidence in your style choices',
  ];

  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-purple-50 via-pink-50 to-amber-50 transition-colors duration-300">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16 sm:py-20 md:py-32">
          <div className="text-center">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 sm:mb-6 bg-gradient-to-r from-purple-600 via-pink-600 to-amber-600 bg-clip-text text-transparent px-2">
              Your AI-Powered Fashion Companion
            </h1>
            <p className="text-lg sm:text-xl md:text-2xl text-gray-700 mb-6 sm:mb-8 max-w-3xl mx-auto px-4">
              Discover what looks best on you with personalized style recommendations based on your unique skin tone and preferences
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/analyze"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-purple-600 text-white font-semibold hover:bg-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                <Camera className="h-5 w-5" />
                <span>Analyze Your Skin Tone</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                href="/recommendations"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-white text-purple-600 font-semibold hover:bg-gray-50 transition-all shadow-lg hover:shadow-xl border-2 border-purple-600"
              >
                <Sparkles className="h-5 w-5" />
                <span>Get Recommendations</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 sm:py-16 md:py-20 bg-gray-50 transition-colors duration-300">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12 sm:mb-16">
            <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4 text-gray-900 px-2">
              Everything You Need to Look Your Best
            </h2>
            <p className="text-base sm:text-lg text-gray-600 max-w-2xl mx-auto px-4">
              Powered by advanced AI and color theory to help you make confident style choices
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 md:gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="p-5 sm:p-6 rounded-2xl bg-white card-hover transition-colors duration-300 border border-gray-200 shadow-sm"
              >
                <div className="w-12 h-12 rounded-full bg-purple-600 flex items-center justify-center mb-4 shadow-lg">
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg sm:text-xl font-bold mb-2 text-gray-900">{feature.title}</h3>
                <p className="text-sm sm:text-base text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-12 sm:py-16 md:py-20 bg-purple-600 transition-colors duration-300">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12 sm:mb-16">
            <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4 text-white px-2">
              How It Works
            </h2>
            <p className="text-base sm:text-lg text-purple-100 max-w-2xl mx-auto px-4">
              Get started in three simple steps
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
            <div className="text-center px-4">
              <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-white text-purple-600 text-xl sm:text-2xl font-bold flex items-center justify-center mx-auto mb-4 shadow-lg">
                1
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2 text-white">Upload Your Photo</h3>
              <p className="text-sm sm:text-base text-purple-100">
                Take a clear selfie or upload an existing photo so we can analyze your skin tone
              </p>
            </div>

            <div className="text-center px-4">
              <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-white text-pink-600 text-xl sm:text-2xl font-bold flex items-center justify-center mx-auto mb-4 shadow-lg">
                2
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2 text-white">Build Your Wardrobe</h3>
              <p className="text-sm sm:text-base text-purple-100">
                Add your clothing items to create a digital wardrobe or explore our suggestions
              </p>
            </div>

            <div className="text-center px-4">
              <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-white text-amber-600 text-xl sm:text-2xl font-bold flex items-center justify-center mx-auto mb-4 shadow-lg">
                3
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2 text-white">Get Recommendations</h3>
              <p className="text-sm sm:text-base text-purple-100">
                Receive personalized outfit suggestions tailored to your unique style and skin tone
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-12 sm:py-16 md:py-20 bg-gray-50 transition-colors duration-300">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-12 items-center">
            <div className="px-2">
              <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-4 sm:mb-6 text-gray-900">
                Why StyleAI?
              </h2>
              <p className="text-base sm:text-lg text-gray-600 mb-6 sm:mb-8">
                StyleAI uses advanced color theory and AI technology to help you look and feel your best every day.
              </p>
              <ul className="space-y-3 sm:space-y-4">
                {benefits.map((benefit, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 sm:h-6 sm:w-6 text-green-500 flex-shrink-0 mt-0.5" />
                    <span className="text-sm sm:text-base text-gray-700">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-100 via-pink-100 to-amber-100 rounded-3xl p-6 sm:p-8 md:p-12 transition-colors duration-300 border border-gray-200">
              <div className="space-y-4 sm:space-y-6">
                <div className="bg-white rounded-2xl p-4 sm:p-6 shadow-lg transition-colors duration-300">
                  <div className="flex items-center gap-3 sm:gap-4 mb-3 sm:mb-4">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
                      <Sparkles className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-sm sm:text-base">AI-Powered</p>
                      <p className="text-xs sm:text-sm text-gray-600">Advanced algorithms</p>
                    </div>
                  </div>
                  <div className="h-2 bg-purple-200 rounded-full overflow-hidden">
                    <div className="h-full w-4/5 bg-purple-600 rounded-full"></div>
                  </div>
                </div>

                <div className="bg-white rounded-2xl p-4 sm:p-6 shadow-lg transition-colors duration-300">
                  <div className="flex items-center gap-3 sm:gap-4 mb-3 sm:mb-4">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-pink-600 flex items-center justify-center flex-shrink-0">
                      <Target className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-sm sm:text-base text-gray-900">Personalized</p>
                      <p className="text-xs sm:text-sm text-gray-600">Just for you</p>
                    </div>
                  </div>
                  <div className="h-2 bg-pink-200 rounded-full overflow-hidden">
                    <div className="h-full w-full bg-pink-600 rounded-full"></div>
                  </div>
                </div>

                <div className="bg-white rounded-2xl p-4 sm:p-6 shadow-lg transition-colors duration-300 border border-gray-100">
                  <div className="flex items-center gap-3 sm:gap-4 mb-3 sm:mb-4">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-amber-600 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-sm sm:text-base text-gray-900">Easy to Use</p>
                      <p className="text-xs sm:text-sm text-gray-600">Simple interface</p>
                    </div>
                  </div>
                  <div className="h-2 bg-amber-200 rounded-full overflow-hidden">
                    <div className="h-full w-11/12 bg-amber-600 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-12 sm:py-16 md:py-20 bg-gradient-to-br from-purple-600 via-pink-600 to-amber-600">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-4 sm:mb-6 px-2">
            Ready to Transform Your Style?
          </h2>
          <p className="text-base sm:text-lg md:text-xl text-white/90 mb-6 sm:mb-8 px-4">
            Join thousands of users who have discovered their perfect style with StyleAI
          </p>
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-6 sm:px-8 py-3 sm:py-4 rounded-full bg-white text-purple-600 font-semibold hover:bg-gray-100 transition-all shadow-lg hover:shadow-xl text-sm sm:text-base"
          >
            <Camera className="h-4 w-4 sm:h-5 sm:w-5" />
            <span>Get Started Now</span>
            <ArrowRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}

