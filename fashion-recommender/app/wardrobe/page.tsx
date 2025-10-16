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
    <div className="min-h-screen section-alt py-12">
      <div className="container-wide">
        {/* Header */}
        <div className="mb-12 animate-fade-in">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-6 mb-8">
            <div>
              <h1 className="heading-primary mb-3 text-foreground">My Wardrobe</h1>
              <p className="text-xl text-muted">
                {filteredItems.length} items in your collection
              </p>
            </div>
            <button
              onClick={() => setShowUploadModal(true)}
              className="btn-primary btn-lg flex items-center gap-3"
            >
              <Plus className="h-6 w-6" />
              <span>Add Items</span>
            </button>
          </div>

          {/* Search and Filter Bar */}
          <div className="flex flex-col md:flex-row gap-6">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-6 w-6 text-muted" />
              <input
                type="text"
                placeholder="Search your wardrobe..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input pl-14 pr-4 py-4 text-lg"
              />
            </div>

            {/* Filter Button */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn-outline btn-lg flex items-center gap-3"
            >
              <Filter className="h-6 w-6" />
              <span>Filters</span>
            </button>

            {/* View Toggle */}
            <div className="flex gap-1 outfit-card p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-3 rounded-lg transition-colors ${
                  viewMode === 'grid' ? 'bg-primary-100 text-primary-600 dark:bg-primary-800 dark:text-primary-300' : 'text-muted hover:text-foreground'
                }`}
              >
                <Grid className="h-6 w-6" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-3 rounded-lg transition-colors ${
                  viewMode === 'list' ? 'bg-primary-100 text-primary-600 dark:bg-primary-800 dark:text-primary-300' : 'text-muted hover:text-foreground'
                }`}
              >
                <List className="h-6 w-6" />
              </button>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-6 outfit-card animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Category Filter */}
                <div>
                  <label className="block text-lg font-semibold text-foreground mb-4">
                    Category
                  </label>
                  <div className="flex flex-wrap gap-3">
                    {categories.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-6 py-3 rounded-full text-sm font-medium transition-colors ${
                          selectedCategory === cat
                            ? 'bg-primary-600 text-white shadow-theme-md'
                            : 'bg-surface text-foreground hover:bg-surface-hover'
                        }`}
                      >
                        {cat}
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
                        key={season}
                        onClick={() => setSelectedSeason(season)}
                        className={`px-6 py-3 rounded-full text-sm font-medium transition-colors ${
                          selectedSeason === season
                            ? 'bg-secondary-600 text-white shadow-theme-md'
                            : 'bg-surface text-foreground hover:bg-surface-hover'
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
          <div className="text-center py-24">
            <div className="w-24 h-24 rounded-2xl bg-surface flex items-center justify-center mx-auto mb-6 shadow-theme-sm">
              <Search className="h-12 w-12 text-muted" />
            </div>
            <h3 className="text-2xl font-semibold text-foreground mb-3">No items found</h3>
            <p className="text-muted mb-8 text-lg">Try adjusting your search or filters</p>
            <button
              onClick={() => {
                setSearchQuery('');
                setSelectedCategory('All');
                setSelectedSeason('All');
              }}
              className="btn-primary btn-lg"
            >
              Clear Filters
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid-responsive animate-fade-in">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="outfit-card overflow-hidden group"
              >
                {/* Item Image */}
                <div
                  className="h-48 flex items-center justify-center text-6xl mb-4 rounded-xl"
                  style={{ backgroundColor: item.color + '20' }}
                >
                  {item.image}
                </div>

                {/* Item Details */}
                <div>
                  <h3 className="font-bold text-xl text-foreground mb-2">{item.name}</h3>
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-muted">{item.category}</span>
                    <span className="text-muted">•</span>
                    <span className="text-muted">{item.season}</span>
                  </div>

                  {/* Color Indicator */}
                  <div className="flex items-center gap-3 mb-6">
                    <div
                      className="w-8 h-8 rounded-full border-2 border-border shadow-sm"
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="text-sm text-muted font-mono">
                      {item.color}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button className="flex-1 btn-ghost py-3 flex items-center justify-center gap-2">
                      <Heart className="h-5 w-5" />
                      <span className="font-medium">Favorite</span>
                    </button>
                    <button className="btn-ghost p-3">
                      <Edit className="h-5 w-5" />
                    </button>
                    <button className="btn-ghost p-3 hover:bg-red-100 hover:text-red-600">
                      <Trash2 className="h-5 w-5" />
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
                className="outfit-card p-6 flex items-center gap-6 hover:shadow-theme-lg transition-all"
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
                  <h3 className="font-bold text-xl text-foreground mb-1">{item.name}</h3>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-sm text-muted">{item.category}</span>
                    <span className="text-muted">•</span>
                    <span className="text-sm text-muted">{item.season}</span>
                    <span className="text-muted">•</span>
                    <div className="flex items-center gap-2">
                      <div
                        className="w-5 h-5 rounded-full border-2 border-border"
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-xs text-muted font-mono">{item.color}</span>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-shrink-0">
                  <button className="px-4 py-2 bg-primary-100 text-primary-600 dark:bg-primary-800 dark:text-primary-300 rounded-lg hover:bg-primary-200 dark:hover:bg-primary-700 transition-colors flex items-center gap-2">
                    <Heart className="h-4 w-4" />
                    <span className="text-sm font-medium">Favorite</span>
                  </button>
                  <button className="btn-ghost p-2">
                    <Edit className="h-4 w-4" />
                  </button>
                  <button className="btn-ghost p-2 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/50 dark:hover:text-red-400">
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
            <div className="outfit-card max-w-2xl w-full p-8 animate-fade-in">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-foreground">Add New Items</h2>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="btn-ghost p-2 rounded-full"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              <div className="border-2 border-dashed border-primary-300 dark:border-primary-600 rounded-xl p-12 text-center mb-6 cursor-pointer hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-950/30 transition-all">
                <Plus className="h-12 w-12 text-primary-600 mx-auto mb-4" />
                <p className="font-semibold text-foreground mb-1">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-muted">
                  PNG, JPG up to 10MB (Supports multiple files)
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="flex-1 btn-ghost py-3 font-medium"
                >
                  Cancel
                </button>
                <button className="flex-1 btn-primary py-3 font-semibold">
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

