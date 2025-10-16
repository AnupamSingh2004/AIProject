import Link from 'next/link';
import { 
  Camera, Shirt, Sparkles, Target, ArrowRight, CheckCircle2, 
  Heart, Palette, Users, Star, Zap, Shield, TrendingUp, Award
} from 'lucide-react';

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
    <div className="min-h-screen animate-fade-in">
      {/* Hero Section */}
      <section className="hero-gradient transition-theme">
        <div className="container-responsive py-20 md:py-32">
          <div className="text-center max-w-5xl mx-auto">
            <div className="mb-6 flex justify-center">
              <div className="h-16 w-16 rounded-2xl fashion-gradient flex items-center justify-center animate-scale-in">
                <Sparkles className="h-8 w-8 text-white" />
              </div>
            </div>
            <h1 className="text-responsive-lg font-bold mb-6 bg-gradient-to-r from-primary-600 via-secondary-600 to-accent-600 bg-clip-text text-transparent animate-slide-up">
              Your AI-Powered Fashion Companion
            </h1>
            <p className="text-responsive-md text-muted mb-8 leading-relaxed max-w-3xl mx-auto animate-slide-up">
              Discover what looks best on you with personalized style recommendations, 
              wardrobe management, and AI-powered fashion insights tailored to your unique style.
            </p>
            
            {/* Feature highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10 max-w-2xl mx-auto">
              <div className="flex items-center justify-center space-x-2 text-sm text-muted">
                <Camera className="h-4 w-4 text-primary-600" />
                <span>AI Skin Tone Analysis</span>
              </div>
              <div className="flex items-center justify-center space-x-2 text-sm text-muted">
                <Shirt className="h-4 w-4 text-secondary-600" />
                <span>Smart Wardrobe Manager</span>
              </div>
              <div className="flex items-center justify-center space-x-2 text-sm text-muted">
                <Target className="h-4 w-4 text-accent-600" />
                <span>Outfit Compatibility</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center animate-scale-in">
              <Link
                href="/analyze"
                className="btn-primary btn-lg inline-flex items-center gap-2 shadow-theme-lg hover:shadow-theme-xl"
              >
                <Camera className="h-5 w-5" />
                <span>Analyze Your Style</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                href="/recommendations"
                className="btn-outline btn-lg inline-flex items-center gap-2 border-2"
              >
                <Sparkles className="h-5 w-5" />
                <span>Get Recommendations</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-surface">
        <div className="container-responsive">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">10K+</div>
              <div className="text-muted font-medium">Happy Users</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-secondary-600 mb-2">50M+</div>
              <div className="text-muted font-medium">Outfit Combinations</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-accent-600 mb-2">98%</div>
              <div className="text-muted font-medium">Color Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">24/7</div>
              <div className="text-muted font-medium">AI Assistant</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container-responsive">
          <div className="text-center mb-16">
            <h2 className="text-responsive-md font-bold mb-4 text-foreground">
              Everything You Need to Look Your Best
            </h2>
            <p className="text-responsive-sm text-muted max-w-2xl mx-auto">
              Powered by advanced AI and color theory to help you make confident style choices
            </p>
          </div>

          <div className="grid-responsive">
            {features.map((feature, index) => {
              const IconComponent = feature.icon
              return (
                <div
                  key={index}
                  className="outfit-card group hover:shadow-theme-md transition-all duration-300 hover:-translate-y-1"
                >
                  <div className="h-16 w-16 rounded-xl fashion-gradient flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                    <IconComponent className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground mb-3">{feature.title}</h3>
                  <p className="text-muted leading-relaxed">{feature.description}</p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 fashion-gradient">
        <div className="container-wide">
          <div className="text-center mb-20">
            <h2 className="heading-primary mb-6 text-white">
              How It Works
            </h2>
            <p className="text-xl text-purple-100 max-w-2xl mx-auto leading-relaxed">
              Get started in three simple steps
            </p>
          </div>

          <div className="grid-responsive md:grid-cols-3">
            <div className="text-center group">
              <div className="h-20 w-20 rounded-2xl bg-white text-purple-600 text-3xl font-bold flex items-center justify-center mx-auto mb-8 shadow-theme-lg group-hover:scale-110 transition-all duration-300">
                1
              </div>
              <h3 className="text-2xl font-bold mb-6 text-white">Upload Your Photo</h3>
              <p className="text-lg text-purple-100 leading-relaxed">
                Take a clear selfie or upload an existing photo so we can analyze your skin tone
              </p>
            </div>

            <div className="text-center group">
              <div className="h-20 w-20 rounded-2xl bg-white text-pink-600 text-3xl font-bold flex items-center justify-center mx-auto mb-8 shadow-theme-lg group-hover:scale-110 transition-all duration-300">
                2
              </div>
              <h3 className="text-2xl font-bold mb-6 text-white">Build Your Wardrobe</h3>
              <p className="text-lg text-purple-100 leading-relaxed">
                Add your clothing items to create a digital wardrobe or explore our suggestions
              </p>
            </div>

            <div className="text-center group">
              <div className="h-20 w-20 rounded-2xl bg-white text-amber-600 text-3xl font-bold flex items-center justify-center mx-auto mb-8 shadow-theme-lg group-hover:scale-110 transition-all duration-300">
                3
              </div>
              <h3 className="text-2xl font-bold mb-6 text-white">Get Recommendations</h3>
              <p className="text-lg text-purple-100 leading-relaxed">
                Receive personalized outfit suggestions tailored to your unique style and skin tone
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="section-default py-20">
        <div className="container-wide">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="heading-primary mb-8 text-foreground">
                Why StyleAI?
              </h2>
              <p className="text-xl text-muted mb-12 leading-relaxed">
                StyleAI uses advanced color theory and AI technology to help you look and feel your best every day.
              </p>
              <ul className="space-y-6">
                {benefits.map((benefit, index) => (
                  <li key={index} className="flex items-start gap-4 group">
                    <CheckCircle2 className="h-6 w-6 text-accent flex-shrink-0 mt-1 group-hover:scale-110 transition-transform" />
                    <span className="text-lg text-foreground leading-relaxed">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="fashion-glow rounded-3xl p-8 lg:p-12">
              <div className="space-y-8">
                <div className="outfit-card hover:scale-105 transition-transform">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="h-16 w-16 rounded-xl bg-purple-600 flex items-center justify-center flex-shrink-0">
                      <Sparkles className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-lg text-foreground">AI-Powered</p>
                      <p className="text-muted">Advanced algorithms</p>
                    </div>
                  </div>
                  <div className="h-3 bg-muted/20 rounded-full overflow-hidden">
                    <div className="h-full w-4/5 bg-purple-600 rounded-full"></div>
                  </div>
                </div>

                <div className="outfit-card hover:scale-105 transition-transform">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="h-16 w-16 rounded-xl bg-pink-600 flex items-center justify-center flex-shrink-0">
                      <Target className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-lg text-foreground">Personalized</p>
                      <p className="text-muted">Just for you</p>
                    </div>
                  </div>
                  <div className="h-3 bg-muted/20 rounded-full overflow-hidden">
                    <div className="h-full w-full bg-pink-600 rounded-full"></div>
                  </div>
                </div>

                <div className="outfit-card hover:scale-105 transition-transform">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="h-16 w-16 rounded-xl bg-amber-600 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <p className="font-bold text-lg text-foreground">Easy to Use</p>
                      <p className="text-muted">Simple interface</p>
                    </div>
                  </div>
                  <div className="h-3 bg-muted/20 rounded-full overflow-hidden">
                    <div className="h-full w-11/12 bg-amber-600 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 fashion-gradient">
        <div className="container-wide text-center">
          <h2 className="heading-primary text-white mb-8">
            Ready to Transform Your Style?
          </h2>
          <p className="text-2xl text-white/90 mb-12 max-w-3xl mx-auto leading-relaxed">
            Join thousands of users who have discovered their perfect style with StyleAI
          </p>
          <Link
            href="/analyze"
            className="cta-button group text-xl"
          >
            <Camera className="h-6 w-6" />
            <span>Get Started Now</span>
            <ArrowRight className="h-6 w-6 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </section>
    </div>
  );
}

