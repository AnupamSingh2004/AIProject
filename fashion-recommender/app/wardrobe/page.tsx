'use client';

import { useState, useEffect } from 'react';
import { Plus, Search, Filter, Grid, List, Trash2, Edit, Heart, X, Upload } from 'lucide-react';
import Image from 'next/image';

interface ClothingItem {
  id: string;
  name: string;
  category: string;
  clothingType: string | null;
  dominantColorR: number | null;
  dominantColorG: number | null;
  dominantColorB: number | null;
  style: string | null;
  pattern: string | null;
  season: string | null;
  occasion: string | null;
  createdAt: Date;
}

const categories = ['All', 'Topwear', 'Bottomwear', 'Dress', 'Footwear', 'Accessories'];
const seasons = ['All', 'Spring', 'Summer', 'Fall', 'Winter'];
const colors = ['All', 'Red', 'Blue', 'Green', 'Black', 'White', 'Gray', 'Brown'];

export default function WardrobePage() {
  const [items, setItems] = useState<ClothingItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedSeason, setSelectedSeason] = useState('All');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  // Get or create userId
  const getUserId = () => {
    if (typeof window === 'undefined') return null;
    let userId = localStorage.getItem('userId');
    if (!userId) {
      userId = crypto.randomUUID();
      localStorage.setItem('userId', userId);
    }
    return userId;
  };

  // Fetch items from database
  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    try {
      setLoading(true);
      const userId = getUserId();
      if (!userId) return;
      
      const response = await fetch(`/api/wardrobe/items?userId=${userId}`);
      const data = await response.json();
      setItems(data.items || []);
    } catch (error) {
      console.error('Error fetching items:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(files);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setUploading(true);
    try {
      const userId = getUserId();
      if (!userId) throw new Error('No user ID');

      // Get or create default wardrobe
      const wardrobeResponse = await fetch('/api/wardrobe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          name: 'My Wardrobe',
        }),
      });
      const wardrobeData = await wardrobeResponse.json();
      const wardrobeId = wardrobeData.wardrobe?.id;

      // Upload each file
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('wardrobeId', wardrobeId);
        formData.append('userId', userId);
        formData.append('name', file.name.replace(/\.[^/.]+$/, ''));
        formData.append('category', 'Topwear'); // Default category

        await fetch('/api/wardrobe/items', {
          method: 'POST',
          body: formData,
        });
      }

      // Refresh items list
      await fetchItems();
      setShowUploadModal(false);
      setSelectedFiles([]);
    } catch (error) {
      console.error('Error uploading items:', error);
      alert('Failed to upload items. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const deleteItem = async (itemId: string) => {
    if (!confirm('Are you sure you want to delete this item?')) return;
    
    try {
      await fetch(`/api/wardrobe/items/${itemId}`, {
        method: 'DELETE',
      });
      await fetchItems();
    } catch (error) {
      console.error('Error deleting item:', error);
    }
  };

  const getColorHex = (item: ClothingItem): string => {
    if (item.dominantColorR !== null && item.dominantColorG !== null && item.dominantColorB !== null) {
      return `#${item.dominantColorR.toString(16).padStart(2, '0')}${item.dominantColorG.toString(16).padStart(2, '0')}${item.dominantColorB.toString(16).padStart(2, '0')}`;
    }
    return '#808080';
  };

  const filteredItems = items.filter((item: ClothingItem) => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || item.category === selectedCategory;
    const matchesSeason = selectedSeason === 'All' || item.season === selectedSeason || !item.season;
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
        {loading ? (
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
            <p className="mt-4 text-muted">Loading your wardrobe...</p>
          </div>
        ) : filteredItems.length === 0 ? (
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
            {filteredItems.map((item: ClothingItem) => (
              <div
                key={item.id}
                className="card p-4 hover:shadow-theme-md transition-all"
              >
                {/* Item Image */}
                <div
                  className="relative h-48 mb-4 rounded-lg overflow-hidden bg-surface"
                >
                  <Image
                    src={`/api/wardrobe/items/${item.id}/image`}
                    alt={item.name}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                </div>

                {/* Item Details */}
                <div>
                  <h3 className="font-semibold text-lg text-foreground mb-2">{item.name}</h3>
                  <div className="flex items-center gap-2 mb-3 text-sm text-muted">
                    <span>{item.category}</span>
                    {item.season && (
                      <>
                        <span>•</span>
                        <span>{item.season}</span>
                      </>
                    )}
                  </div>

                  {/* Color Indicator */}
                  <div className="flex items-center gap-2 mb-4">
                    <div
                      className="w-6 h-6 rounded-full border-2 border-border"
                      style={{ backgroundColor: getColorHex(item) }}
                    ></div>
                    <span className="text-xs text-muted font-mono">
                      {getColorHex(item)}
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
                    <button 
                      onClick={() => deleteItem(item.id)}
                      className="btn btn-ghost btn-sm hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-3 animate-fade-in">
            {filteredItems.map((item: ClothingItem) => (
              <div
                key={item.id}
                className="card p-4 flex items-center gap-4 hover:shadow-theme-md transition-all"
              >
                {/* Item Preview */}
                <div className="relative w-20 h-20 rounded-lg overflow-hidden bg-surface flex-shrink-0">
                  <Image
                    src={`/api/wardrobe/items/${item.id}/image`}
                    alt={item.name}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                </div>

                {/* Item Details */}
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-foreground mb-1">{item.name}</h3>
                  <div className="flex items-center gap-2 text-sm text-muted">
                    <span>{item.category}</span>
                    {item.season && (
                      <>
                        <span>•</span>
                        <span>{item.season}</span>
                      </>
                    )}
                    <span>•</span>
                    <div className="flex items-center gap-1">
                      <div
                        className="w-4 h-4 rounded-full border border-border"
                        style={{ backgroundColor: getColorHex(item) }}
                      ></div>
                      <span className="text-xs font-mono">{getColorHex(item)}</span>
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
                  <button 
                    onClick={() => deleteItem(item.id)}
                    className="btn-ghost p-2 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30"
                  >
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
                  onClick={() => {
                    setShowUploadModal(false);
                    setSelectedFiles([]);
                  }}
                  className="btn-ghost p-2"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="border-2 border-dashed border-primary-300 dark:border-primary-600 rounded-lg p-8 text-center mb-6 cursor-pointer hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-950/20 transition-all">
                <input
                  type="file"
                  id="file-upload"
                  multiple
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="h-10 w-10 text-primary-600 mx-auto mb-3" />
                  <p className="font-medium text-foreground mb-1">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-muted">
                    PNG, JPG up to 10MB (Multiple files supported)
                  </p>
                  {selectedFiles.length > 0 && (
                    <p className="text-sm text-primary-600 mt-2 font-medium">
                      {selectedFiles.length} file(s) selected
                    </p>
                  )}
                </label>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setShowUploadModal(false);
                    setSelectedFiles([]);
                  }}
                  className="flex-1 btn-ghost"
                  disabled={uploading}
                >
                  Cancel
                </button>
                <button 
                  onClick={handleUpload}
                  className="flex-1 btn-primary"
                  disabled={uploading || selectedFiles.length === 0}
                >
                  {uploading ? 'Uploading...' : 'Upload Items'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

