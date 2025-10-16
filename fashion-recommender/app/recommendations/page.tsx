'use client';

import { useState } from 'react';
import { Sparkles, RefreshCw, Heart, Info, Filter, Sun, Snowflake, Leaf, Flower2 } from 'lucide-react';

// Mock recommendations data
const mockRecommendations = [
  {
    id: 1,
    items: ['👕', '👖', '👟'],
    title: 'Casual Day Out',
    score: 95,
    colors: ['#4A90E2', '#2C3E50', '#FFFFFF'],
    explanation: 'Perfect color harmony with your warm undertone. Blue and earth tones complement your skin beautifully.',
    occasion: 'Casual',
    season: 'Spring',
  },
  {
    id: 2,
    items: ['👔', '👞', '🧥'],
    title: 'Business Professional',
    score: 92,
    colors: ['#2C3E50', '#8B4513', '#34495E'],
    explanation: 'Classic combination that suits your skin tone. Navy and brown create a sophisticated look.',
    occasion: 'Business',
    season: 'All',
  },
  {
    id: 3,
    items: ['👗', '👠', '👜'],
    title: 'Evening Elegance',
    score: 88,
    colors: ['#E74C3C', '#FFD700', '#2C3E50'],
    explanation: 'Bold colors that work beautifully with your warm undertone. Red is particularly flattering.',
    occasion: 'Formal',
    season: 'Summer',
  },
  {
    id: 4,
    items: ['🧥', '👚', '👖'],
    title: 'Autumn Comfort',
    score: 90,
    colors: ['#D4A574', '#8B4513', '#556B2F'],
    explanation: 'Earthy autumn tones enhance your natural warmth. Perfect seasonal palette.',
    occasion: 'Casual',
    season: 'Fall',
  },
  {
    id: 5,
    items: ['👕', '🩳', '🕶️'],
    title: 'Summer Vibes',
    score: 87,
    colors: ['#FFA500', '#FFFFFF', '#4682B4'],
    explanation: 'Light and airy summer colors that complement your skin tone beautifully.',
    occasion: 'Casual',
    season: 'Summer',
  },
  {
    id: 6,
    items: ['🧥', '👖', '🧣'],
    title: 'Winter Warmth',
    score: 91,
    colors: ['#8B0000', '#000080', '#DCDCDC'],
    explanation: 'Rich winter colors that create depth and warmth against your skin tone.',
    occasion: 'Casual',
    season: 'Winter',
  },
];

const occasions = ['All', 'Casual', 'Business', 'Formal', 'Party', 'Sports'];
const seasons = [
  { name: 'All', icon: null },
  { name: 'Spring', icon: Flower2 },
  { name: 'Summer', icon: Sun },
  { name: 'Fall', icon: Leaf },
  { name: 'Winter', icon: Snowflake },
];

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState(mockRecommendations);
  const [selectedOccasion, setSelectedOccasion] = useState('All');
  const [selectedSeason, setSelectedSeason] = useState('All');
  const [showFilters, setShowFilters] = useState(false);
  const [favorites, setFavorites] = useState<number[]>([]);

  const filteredRecommendations = recommendations.filter((rec) => {
    const matchesOccasion = selectedOccasion === 'All' || rec.occasion === selectedOccasion;
    const matchesSeason = selectedSeason === 'All' || rec.season === selectedSeason || rec.season === 'All';
    return matchesOccasion && matchesSeason;
  });

  const toggleFavorite = (id: number) => {
    setFavorites((prev) =>
      prev.includes(id) ? prev.filter((fav) => fav !== id) : [...prev, id]
    );
  };

  const refreshRecommendations = () => {
    // Simulate refreshing recommendations
    const shuffled = [...recommendations].sort(() => Math.random() - 0.5);
    setRecommendations(shuffled);
  };

  return (
    <div className="min-h-screen section-alt py-12">
      <div className="container-wide">
        {/* Header */}
        <div className="mb-12 animate-fade-in">
          <div className="flex-responsive justify-between items-start sm:items-center mb-8">
            <div>
              <h1 className="text-responsive-md text-foreground mb-3 flex items-center gap-3 font-bold">
                <Sparkles className="h-8 w-8 sm:h-12 sm:w-12 text-primary-600" />
                <span>Recommendations</span>
              </h1>
              <p className="text-responsive-xs text-muted">
                Personalized outfits curated just for you
              </p>
            </div>
            <button
              onClick={refreshRecommendations}
              className="btn btn-primary btn-responsive flex items-center gap-3 justify-center"
            >
              <RefreshCw className="h-5 w-5" />
              <span>Refresh</span>
            </button>
          </div>

          {/* Today's Pick Banner */}
          <div className="fashion-gradient rounded-2xl p-10 text-white mb-8 shadow-theme-lg">
            <h2 className="text-3xl font-bold mb-3">Today's Featured Outfit</h2>
            <p className="text-white/90 mb-4">Based on your preferences and the season</p>
            <div className="flex items-center gap-6">
              <div className="flex gap-4 text-6xl">
                {mockRecommendations[0].items.map((item, idx) => (
                  <span key={idx}>{item}</span>
                ))}
              </div>
              <div className="flex-1">
                <p className="text-lg font-semibold mb-2">
                  {mockRecommendations[0].title}
                </p>
                <p className="text-white/80 text-sm">
                  {mockRecommendations[0].explanation}
                </p>
              </div>
              <button className="btn btn-outline btn-md">
                View Details
              </button>
            </div>
          </div>

          {/* Filter Bar */}
          <div className="flex-responsive gap-4">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn btn-outline btn-responsive flex items-center gap-2 justify-center"
            >
              <Filter className="h-4 w-4" />
              <span>Filters</span>
            </button>

            <div className="flex-1 flex flex-wrap gap-2 items-center min-w-0">
              <span className="px-2 py-1 text-xs sm:text-sm font-medium text-muted hidden sm:flex items-center">
                Quick:
              </span>
              {['Casual', 'Business', 'Formal'].map((occasion) => (
                <button
                  key={occasion}
                  onClick={() => setSelectedOccasion(occasion)}
                  className={`btn btn-sm ${
                    selectedOccasion === occasion
                      ? 'btn-primary'
                      : 'btn-secondary'
                  }`}
                >
                  {occasion}
                </button>
              ))}
            </div>
          </div>

          {/* Expanded Filters */}
          {showFilters && (
            <div className="mt-6 outfit-card animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Occasion Filter */}
                <div>
                  <label className="block text-lg font-semibold text-foreground mb-4">
                    Occasion
                  </label>
                  <div className="flex flex-wrap gap-3">
                    {occasions.map((occasion) => (
                      <button
                        key={occasion}
                        onClick={() => setSelectedOccasion(occasion)}
                        className={`btn btn-md ${
                          selectedOccasion === occasion
                            ? 'btn-primary'
                            : 'btn-secondary'
                        }`}
                      >
                        {occasion}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Season Filter */}
                <div>
                  <label className="block text-lg font-semibold text-foreground mb-4">
                    Season
                  </label>
                  <div className="flex flex-wrap gap-3">
                    {seasons.map((season) => (
                      <button
                        key={season.name}
                        onClick={() => setSelectedSeason(season.name)}
                        className={`btn btn-md flex items-center gap-2 ${
                          selectedSeason === season.name
                            ? 'btn-secondary'
                            : 'btn-outline'
                        }`}
                      >
                        {season.icon && <season.icon className="h-4 w-4" />}
                        <span>{season.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recommendations Grid */}
        {filteredRecommendations.length === 0 ? (
          <div className="text-center py-24">
            <div className="w-24 h-24 rounded-2xl bg-surface flex items-center justify-center mx-auto mb-6 shadow-theme-sm">
              <Sparkles className="h-12 w-12 text-muted" />
            </div>
            <h3 className="text-2xl font-semibold text-foreground mb-3">No recommendations found</h3>
            <p className="text-muted mb-8 text-lg">Try adjusting your filters</p>
            <button
              onClick={() => {
                setSelectedOccasion('All');
                setSelectedSeason('All');
              }}
              className="btn btn-primary btn-lg"
            >
              Clear Filters
            </button>
          </div>
        ) : (
          <div className="grid-responsive animate-fade-in">
            {filteredRecommendations.map((rec) => (
              <div
                key={rec.id}
                className="outfit-card overflow-hidden group"
              >
                {/* Score Badge */}
                <div className="relative">
                  <div className="fashion-gradient p-8 flex justify-center items-center gap-4">
                    <div className="flex gap-3 text-5xl">
                      {rec.items.map((item, idx) => (
                        <span key={idx}>{item}</span>
                      ))}
                    </div>
                  </div>
                  <div className="absolute top-4 right-4 px-3 py-2 bg-green-500 text-white rounded-full text-sm font-bold shadow-theme-md">
                    {rec.score}% Match
                  </div>
                  <button
                    onClick={() => toggleFavorite(rec.id)}
                    className="btn btn-ghost btn-sm absolute top-4 left-4"
                  >
                    <Heart
                      className={`h-5 w-5 ${
                        favorites.includes(rec.id)
                          ? 'fill-red-500 text-red-500'
                          : 'text-muted'
                      }`}
                    />
                  </button>
                </div>

                {/* Details */}
                <div className="p-6">
                  <h3 className="text-xl font-bold text-foreground mb-3">{rec.title}</h3>
                  
                  {/* Tags */}
                  <div className="flex gap-3 mb-4">
                    <span className="px-4 py-2 bg-primary-100 text-primary-700 dark:bg-primary-800 dark:text-primary-300 rounded-full text-sm font-medium">
                      {rec.occasion}
                    </span>
                    <span className="px-4 py-2 bg-secondary-100 text-secondary-700 dark:bg-secondary-800 dark:text-secondary-300 rounded-full text-sm font-medium">
                      {rec.season}
                    </span>
                  </div>

                  {/* Color Palette */}
                  <div className="flex gap-3 mb-4">
                    {rec.colors.map((color, idx) => (
                      <div
                        key={idx}
                        className="w-8 h-8 rounded-full border-2 border-border shadow-theme-sm"
                        style={{ backgroundColor: color }}
                      ></div>
                    ))}
                  </div>

                  {/* Explanation */}
                  <div className="p-4 bg-surface rounded-xl mb-6">
                    <div className="flex items-start gap-3">
                      <Info className="h-5 w-5 text-primary-600 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-muted leading-relaxed">{rec.explanation}</p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-3">
                    <button className="btn btn-primary btn-md flex-1">
                      Try It On
                    </button>
                    <button className="btn btn-ghost btn-md flex-1">
                      Save
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Info Banner */}
        <div className="mt-12 outfit-card">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-primary-100 dark:bg-primary-800 rounded-xl">
              <Info className="h-6 w-6 text-primary-600 dark:text-primary-300" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-2 text-lg">
                How are these recommendations generated?
              </h3>
              <p className="text-muted leading-relaxed">
                Our AI analyzes your skin tone, undertone, and personal style preferences to suggest
                outfits that complement your natural coloring. We use advanced color theory and fashion
                principles to ensure every recommendation looks great on you.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

