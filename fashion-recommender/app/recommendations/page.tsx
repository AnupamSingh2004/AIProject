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
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2 flex items-center gap-3">
                <Sparkles className="h-10 w-10 text-purple-600" />
                Recommendations
              </h1>
              <p className="text-gray-600">
                Personalized outfits curated just for you
              </p>
            </div>
            <button
              onClick={refreshRecommendations}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors shadow-lg"
            >
              <RefreshCw className="h-5 w-5" />
              <span>Refresh</span>
            </button>
          </div>

          {/* Today's Pick Banner */}
          <div className="bg-gradient-to-r from-purple-600 via-pink-600 to-amber-600 rounded-2xl p-8 text-white mb-6">
            <h2 className="text-2xl font-bold mb-2">Today's Featured Outfit</h2>
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
              <button className="px-6 py-3 bg-white text-purple-600 rounded-full font-semibold hover:bg-gray-100 transition-colors">
                View Details
              </button>
            </div>
          </div>

          {/* Filter Bar */}
          <div className="flex flex-col md:flex-row gap-4">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 px-6 py-3 bg-white border border-gray-300 text-gray-900 rounded-xl hover:bg-gray-50 transition-colors"
            >
              <Filter className="h-5 w-5" />
              <span>Filters</span>
            </button>

            <div className="flex-1 flex flex-wrap gap-2">
              <span className="px-4 py-3 text-sm font-medium text-gray-700 flex items-center">
                Quick Filters:
              </span>
              {['Casual', 'Business', 'Formal'].map((occasion) => (
                <button
                  key={occasion}
                  onClick={() => setSelectedOccasion(occasion)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                    selectedOccasion === occasion
                      ? 'bg-purple-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
                  }`}
                >
                  {occasion}
                </button>
              ))}
            </div>
          </div>

          {/* Expanded Filters */}
          {showFilters && (
            <div className="mt-4 p-6 bg-white border rounded-xl shadow-md animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Occasion Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Occasion
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {occasions.map((occasion) => (
                      <button
                        key={occasion}
                        onClick={() => setSelectedOccasion(occasion)}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                          selectedOccasion === occasion
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {occasion}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Season Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Season
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {seasons.map((season) => (
                      <button
                        key={season.name}
                        onClick={() => setSelectedSeason(season.name)}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-colors flex items-center gap-2 ${
                          selectedSeason === season.name
                            ? 'bg-pink-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
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
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
              <Sparkles className="h-10 w-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              No recommendations found
            </h3>
            <p className="text-gray-600 mb-6">Try adjusting your filters</p>
            <button
              onClick={() => {
                setSelectedOccasion('All');
                setSelectedSeason('All');
              }}
              className="px-6 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors"
            >
              Clear Filters
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
            {filteredRecommendations.map((rec) => (
              <div
                key={rec.id}
                className="bg-white border rounded-2xl shadow-md overflow-hidden card-hover"
              >
                {/* Score Badge */}
                <div className="relative">
                  <div className="bg-gradient-to-br from-purple-100 to-pink-100 p-8 flex justify-center items-center gap-4">
                    <div className="flex gap-3 text-5xl">
                      {rec.items.map((item, idx) => (
                        <span key={idx}>{item}</span>
                      ))}
                    </div>
                  </div>
                  <div className="absolute top-4 right-4 px-3 py-1 bg-green-500 text-white rounded-full text-sm font-bold">
                    {rec.score}% Match
                  </div>
                  <button
                    onClick={() => toggleFavorite(rec.id)}
                    className="absolute top-4 left-4 p-2 bg-white rounded-full shadow-lg hover:bg-gray-50 transition-colors"
                  >
                    <Heart
                      className={`h-5 w-5 ${
                        favorites.includes(rec.id)
                          ? 'fill-red-500 text-red-500'
                          : 'text-gray-600'
                      }`}
                    />
                  </button>
                </div>

                {/* Details */}
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{rec.title}</h3>
                  
                  {/* Tags */}
                  <div className="flex gap-2 mb-3">
                    <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                      {rec.occasion}
                    </span>
                    <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                      {rec.season}
                    </span>
                  </div>

                  {/* Color Palette */}
                  <div className="flex gap-2 mb-4">
                    {rec.colors.map((color, idx) => (
                      <div
                        key={idx}
                        className="w-8 h-8 rounded-full border-2 border-white shadow-md"
                        style={{ backgroundColor: color }}
                      ></div>
                    ))}
                  </div>

                  {/* Explanation */}
                  <div className="p-3 bg-gray-50 rounded-lg mb-4">
                    <div className="flex items-start gap-2">
                      <Info className="h-4 w-4 text-purple-600 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-gray-700">{rec.explanation}</p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button className="flex-1 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium">
                      Try It On
                    </button>
                    <button className="flex-1 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium">
                      Save
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Info Banner */}
        <div className="mt-12 bg-blue-50 border border-blue-200 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <Info className="h-6 w-6 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900 mb-1">
                How are these recommendations generated?
              </h3>
              <p className="text-sm text-blue-800">
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

