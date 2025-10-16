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
          <div className="flex-responsive justify-between items-start sm:items-center mb-8">
            <div>
              <h1 className="text-responsive-md mb-3 text-foreground font-bold">My Wardrobe</h1>
              <p className="text-responsive-xs text-muted">
                {filteredItems.length} items in your collection
              </p>
            </div>
            <button
              onClick={() => setShowUploadModal(true)}
              className="btn btn-primary btn-responsive"
            >
              <Plus className="h-5 w-5" />
              <span>Add Items</span>
            </button>
          </div>

          {/* Search and Filter Bar */}
          <div className="flex-responsive gap-4">
            {/* Search */}
            <div className="flex-1 relative min-w-0">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted" />
              <input
                type="text"
                placeholder="Search wardrobe..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input pl-10 pr-4 py-3 w-full"
              />
            </div>

            <div className="flex gap-2 flex-shrink-0">
              {/* Filter Button */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="btn btn-outline btn-md"
              >
                <Filter className="h-4 w-4" />
                <span className="hidden sm:inline">Filters</span>
              </button>

              {/* View Toggle */}
              <div className="flex gap-1 bg-surface rounded-lg p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-md transition-colors ${
                    viewMode === 'grid' ? 'bg-primary-500 text-white' : 'text-muted hover:text-foreground'
                  }`}
                  aria-label="Grid view"
                >
                  <Grid className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-md transition-colors ${
                    viewMode === 'list' ? 'bg-primary-500 text-white' : 'text-muted hover:text-foreground'
                  }`}
                  aria-label="List view"
                >
                  <List className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-4 card p-6 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Category Filter */}
                <div>
                  <label className="block text-sm font-medium text-foreground mb-3">
                    Category
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {categories.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                          selectedCategory === cat
                            ? 'bg-primary-500 text-white'
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
                  <label className="block text-sm font-medium text-foreground mb-3">
                    Season
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {seasons.map((season) => (
                      <button
                        key={season}
                        onClick={() => setSelectedSeason(season)}
                        className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                          selectedSeason === season
                            ? 'bg-secondary-500 text-white'
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
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-lg bg-surface flex items-center justify-center mx-auto mb-4">
              <Search className="h-8 w-8 text-muted" />
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">No items found</h3>
            <p className="text-muted mb-6">Try adjusting your search or filters</p>
            <button
              onClick={() => {
                setSearchQuery('');
                setSelectedCategory('All');
                setSelectedSeason('All');
              }}
              className="btn btn-primary"
            >
              Clear Filters
            </button>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid-responsive animate-fade-in">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="card p-4 hover:shadow-theme-md transition-all"
              >
                {/* Item Image */}
                <div
                  className="h-40 flex items-center justify-center text-5xl mb-4 rounded-lg"
                  style={{ backgroundColor: item.color + '20' }}
                >
                  {item.image}
                </div>

                {/* Item Details */}
                <div>
                  <h3 className="font-semibold text-lg text-foreground mb-2">{item.name}</h3>
                  <div className="flex items-center gap-2 mb-3 text-sm text-muted">
                    <span>{item.category}</span>
                    <span>•</span>
                    <span>{item.season}</span>
                  </div>

                  {/* Color Indicator */}
                  <div className="flex items-center gap-2 mb-4">
                    <div
                      className="w-6 h-6 rounded-full border-2 border-border"
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="text-xs text-muted font-mono">
                      {item.color}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button className="btn btn-ghost btn-sm flex-1">
                      <Heart className="h-4 w-4" />
                      <span>Favorite</span>
                    </button>
                    <button className="btn btn-ghost btn-sm">
                      <Edit className="h-4 w-4" />
                    </button>
                    <button className="btn btn-ghost btn-sm hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-3 animate-fade-in">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="card p-4 flex items-center gap-4 hover:shadow-theme-md transition-all"
              >
                {/* Item Preview */}
                <div
                  className="w-16 h-16 rounded-lg flex items-center justify-center text-2xl flex-shrink-0"
                  style={{ backgroundColor: item.color + '20' }}
                >
                  {item.image}
                </div>

                {/* Item Details */}
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-foreground mb-1">{item.name}</h3>
                  <div className="flex items-center gap-2 text-sm text-muted">
                    <span>{item.category}</span>
                    <span>•</span>
                    <span>{item.season}</span>
                    <span>•</span>
                    <div className="flex items-center gap-1">
                      <div
                        className="w-4 h-4 rounded-full border border-border"
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-xs font-mono">{item.color}</span>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 flex-shrink-0">
                  <button className="px-3 py-2 bg-primary-100 text-primary-600 dark:bg-primary-800 dark:text-primary-300 rounded-md hover:bg-primary-200 dark:hover:bg-primary-700 transition-colors flex items-center gap-2 text-sm">
                    <Heart className="h-4 w-4" />
                    <span>Favorite</span>
                  </button>
                  <button className="btn-ghost p-2">
                    <Edit className="h-4 w-4" />
                  </button>
                  <button className="btn-ghost p-2 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30">
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
            <div className="card max-w-lg w-full p-6 animate-fade-in">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-foreground">Add New Items</h2>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="btn-ghost p-2"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="border-2 border-dashed border-primary-300 dark:border-primary-600 rounded-lg p-8 text-center mb-6 cursor-pointer hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-950/20 transition-all">
                <Plus className="h-10 w-10 text-primary-600 mx-auto mb-3" />
                <p className="font-medium text-foreground mb-1">
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-muted">
                  PNG, JPG up to 10MB (Multiple files supported)
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="flex-1 btn-ghost"
                >
                  Cancel
                </button>
                <button className="flex-1 btn-primary">
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

