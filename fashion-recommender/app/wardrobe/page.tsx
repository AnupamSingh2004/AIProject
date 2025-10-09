'use client';

import { useState } from 'react';
import { Plus, Search, Filter, Grid, List, Trash2, Edit, Heart, X } from 'lucide-react';

// Mock data for demonstration
const mockItems = [
  { id: 1, name: 'Blue Denim Jacket', category: 'Outerwear', color: '#4A90E2', season: 'Fall', image: '🧥' },
  { id: 2, name: 'White T-Shirt', category: 'Tops', color: '#FFFFFF', season: 'All', image: '👕' },
  { id: 3, name: 'Black Jeans', category: 'Bottoms', color: '#2C3E50', season: 'All', image: '👖' },
  { id: 4, name: 'Red Dress', category: 'Dresses', color: '#E74C3C', season: 'Summer', image: '👗' },
  { id: 5, name: 'Gray Sweater', category: 'Tops', color: '#95A5A6', season: 'Winter', image: '👚' },
  { id: 6, name: 'Brown Boots', category: 'Shoes', color: '#8B4513', season: 'Fall', image: '👢' },
  { id: 7, name: 'Green Blazer', category: 'Outerwear', color: '#27AE60', season: 'Spring', image: '🧥' },
  { id: 8, name: 'Floral Skirt', category: 'Bottoms', color: '#F39C12', season: 'Spring', image: '👗' },
];

const categories = ['All', 'Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Accessories'];
const seasons = ['All', 'Spring', 'Summer', 'Fall', 'Winter'];
const colors = ['All', 'Red', 'Blue', 'Green', 'Black', 'White', 'Gray', 'Brown'];

export default function WardrobePage() {
  const [items, setItems] = useState(mockItems);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedSeason, setSelectedSeason] = useState('All');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  const filteredItems = items.filter((item) => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || item.category === selectedCategory;
    const matchesSeason = selectedSeason === 'All' || item.season === selectedSeason || item.season === 'All';
    return matchesSearch && matchesCategory && matchesSeason;
  });

  return (
    <div className="min-h-screen bg-gray-50 py-8 transition-colors duration-300">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
            <div>
              <h1 className="text-4xl font-bold mb-2 text-gray-900">My Wardrobe</h1>
              <p className="text-gray-600">
                {filteredItems.length} items in your collection
              </p>
            </div>
            <button
              onClick={() => setShowUploadModal(true)}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors shadow-lg"
            >
              <Plus className="h-5 w-5" />
              <span>Add Items</span>
            </button>
          </div>

          {/* Search and Filter Bar */}
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search your wardrobe..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-500 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            {/* Filter Button */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 px-6 py-3 bg-white border border-gray-300 text-gray-900 rounded-xl hover:bg-gray-50 transition-colors"
            >
              <Filter className="h-5 w-5" />
              <span>Filters</span>
            </button>

            {/* View Toggle */}
            <div className="flex gap-2 bg-white border border-gray-300 rounded-xl p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'grid' ? 'bg-purple-100 text-purple-600' : 'text-gray-600'
                }`}
              >
                <Grid className="h-5 w-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'list' ? 'bg-purple-100 text-purple-600' : 'text-gray-600'
                }`}
              >
                <List className="h-5 w-5" />
              </button>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-4 p-6 bg-white border border-gray-200 rounded-xl shadow-md animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Category Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Category
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {categories.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                          selectedCategory === cat
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Season Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Season
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {seasons.map((season) => (
                      <button
                        key={season}
                        onClick={() => setSelectedSeason(season)}
                        className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                          selectedSeason === season
                            ? 'bg-pink-600 text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {season}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Items Grid/List */}
        {filteredItems.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
              <Search className="h-10 w-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No items found</h3>
            <p className="text-gray-600 mb-6">Try adjusting your search or filters</p>
            <button
              onClick={() => {
                setSearchQuery('');
                setSelectedCategory('All');
                setSelectedSeason('All');
              }}
              className="px-6 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors"
            >
              Clear Filters
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-fade-in">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="bg-white border border-gray-200 rounded-2xl shadow-md overflow-hidden card-hover"
              >
                {/* Item Image */}
                <div
                  className="h-48 flex items-center justify-center text-6xl"
                  style={{ backgroundColor: item.color + '20' }}
                >
                  {item.image}
                </div>

                {/* Item Details */}
                <div className="p-4">
                  <h3 className="font-bold text-lg text-gray-900 mb-1">{item.name}</h3>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-sm text-gray-600">{item.category}</span>
                    <span className="text-gray-300">•</span>
                    <span className="text-sm text-gray-600">{item.season}</span>
                  </div>

                  {/* Color Indicator */}
                  <div className="flex items-center gap-2 mb-4">
                    <div
                      className="w-6 h-6 rounded-full border-2 border-gray-200"
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="text-xs text-gray-500">
                      {item.color}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button className="flex-1 py-2 bg-purple-100 text-purple-600 rounded-lg hover:bg-purple-200 transition-colors flex items-center justify-center gap-1">
                      <Heart className="h-4 w-4" />
                      <span className="text-sm font-medium">Favorite</span>
                    </button>
                    <button className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors">
                      <Edit className="h-4 w-4" />
                    </button>
                    <button className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-colors">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-4 animate-fade-in">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="bg-white border border-gray-200 rounded-xl shadow-md p-6 flex items-center gap-6 hover:shadow-lg transition-all"
              >
                {/* Item Preview */}
                <div
                  className="w-24 h-24 rounded-lg flex items-center justify-center text-4xl flex-shrink-0"
                  style={{ backgroundColor: item.color + '20' }}
                >
                  {item.image}
                </div>

                {/* Item Details */}
                <div className="flex-1">
                  <h3 className="font-bold text-xl text-gray-900 mb-1">{item.name}</h3>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-sm text-gray-600">{item.category}</span>
                    <span className="text-gray-300">•</span>
                    <span className="text-sm text-gray-600">{item.season}</span>
                    <span className="text-gray-300">•</span>
                    <div className="flex items-center gap-2">
                      <div
                        className="w-5 h-5 rounded-full border-2 border-gray-200"
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-xs text-gray-500">{item.color}</span>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-shrink-0">
                  <button className="px-4 py-2 bg-purple-100 text-purple-600 rounded-lg hover:bg-purple-200 transition-colors flex items-center gap-2">
                    <Heart className="h-4 w-4" />
                    <span className="text-sm font-medium">Favorite</span>
                  </button>
                  <button className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors">
                    <Edit className="h-4 w-4" />
                  </button>
                  <button className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition-colors">
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Upload Modal */}
        {showUploadModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white border rounded-2xl max-w-2xl w-full p-8 animate-fade-in">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Add New Items</h2>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="p-2 hover:bg-gray-100 text-gray-600 rounded-full transition-colors"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              <div className="border-2 border-dashed border-purple-300 rounded-xl p-12 text-center mb-6 cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-all">
                <Plus className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <p className="font-semibold text-gray-900 mb-1">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-600">
                  PNG, JPG up to 10MB (Supports multiple files)
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="flex-1 py-3 bg-gray-100 text-gray-700 rounded-xl font-medium hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button className="flex-1 py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors">
                  Upload Items
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

