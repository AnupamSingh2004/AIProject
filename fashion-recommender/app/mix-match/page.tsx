'use client';

import { useState } from 'react';
import { Shuffle, Save, Share2, AlertCircle, CheckCircle2, XCircle, Info } from 'lucide-react';

// Mock wardrobe items for mix and match
const wardrobeItems = {
  tops: [
    { id: 1, name: 'White T-Shirt', color: '#FFFFFF', icon: '👕' },
    { id: 2, name: 'Blue Shirt', color: '#4A90E2', icon: '👔' },
    { id: 3, name: 'Gray Sweater', color: '#95A5A6', icon: '👚' },
    { id: 4, name: 'Black Turtleneck', color: '#2C3E50', icon: '👕' },
  ],
  bottoms: [
    { id: 5, name: 'Black Jeans', color: '#2C3E50', icon: '👖' },
    { id: 6, name: 'Blue Jeans', color: '#4A90E2', icon: '👖' },
    { id: 7, name: 'Beige Chinos', color: '#D4A574', icon: '👖' },
    { id: 8, name: 'Black Skirt', color: '#000000', icon: '👗' },
  ],
  shoes: [
    { id: 9, name: 'White Sneakers', color: '#FFFFFF', icon: '👟' },
    { id: 10, name: 'Brown Boots', color: '#8B4513', icon: '👢' },
    { id: 11, name: 'Black Heels', color: '#000000', icon: '👠' },
    { id: 12, name: 'Loafers', color: '#654321', icon: '👞' },
  ],
  accessories: [
    { id: 13, name: 'Gold Necklace', color: '#FFD700', icon: '📿' },
    { id: 14, name: 'Black Bag', color: '#000000', icon: '👜' },
    { id: 15, name: 'Sunglasses', color: '#000000', icon: '🕶️' },
    { id: 16, name: 'Watch', color: '#C0C0C0', icon: '⌚' },
  ],
};

type SelectedItems = {
  top: any | null;
  bottom: any | null;
  shoes: any | null;
  accessory: any | null;
};

export default function MixMatchPage() {
  const [selectedItems, setSelectedItems] = useState<SelectedItems>({
    top: null,
    bottom: null,
    shoes: null,
    accessory: null,
  });

  const selectItem = (category: keyof SelectedItems, item: any) => {
    setSelectedItems((prev) => ({
      ...prev,
      [category]: prev[category]?.id === item.id ? null : item,
    }));
  };

  const calculateCompatibility = () => {
    const selected = Object.values(selectedItems).filter((item) => item !== null);
    if (selected.length < 2) return null;

    // Simple mock compatibility calculation
    const hasTop = selectedItems.top !== null;
    const hasBottom = selectedItems.bottom !== null;
    const hasShoes = selectedItems.shoes !== null;

    let score = 60 + Math.random() * 30; // Random score between 60-90
    
    if (hasTop && hasBottom) score += 5;
    if (hasShoes) score += 3;

    return Math.round(score);
  };

  const compatibility = calculateCompatibility();

  const getCompatibilityColor = (score: number | null) => {
    if (!score) return 'gray';
    if (score >= 80) return 'green';
    if (score >= 65) return 'yellow';
    return 'red';
  };

  const getCompatibilityMessage = (score: number | null) => {
    if (!score) return 'Select at least 2 items to see compatibility';
    if (score >= 80) return 'Excellent combination! This outfit looks great together.';
    if (score >= 65) return 'Good combination with minor improvements possible.';
    return 'This combination needs adjustment. Consider changing some items.';
  };

  const randomizeOutfit = () => {
    const randomTop = wardrobeItems.tops[Math.floor(Math.random() * wardrobeItems.tops.length)];
    const randomBottom = wardrobeItems.bottoms[Math.floor(Math.random() * wardrobeItems.bottoms.length)];
    const randomShoes = wardrobeItems.shoes[Math.floor(Math.random() * wardrobeItems.shoes.length)];
    const randomAccessory = wardrobeItems.accessories[Math.floor(Math.random() * wardrobeItems.accessories.length)];

    setSelectedItems({
      top: randomTop,
      bottom: randomBottom,
      shoes: randomShoes,
      accessory: randomAccessory,
    });
  };

  const clearAll = () => {
    setSelectedItems({
      top: null,
      bottom: null,
      shoes: null,
      accessory: null,
    });
  };

  const compatibilityColor = getCompatibilityColor(compatibility);

  return (
    <div className="min-h-screen section-alt py-12">
      <div className="container-wide">
        {/* Header */}
        <div className="mb-12 animate-fade-in">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-6 mb-8">
            <div>
              <h1 className="heading-primary text-foreground mb-3">Mix & Match</h1>
              <p className="text-xl text-muted">
                Create and test outfit combinations from your wardrobe
              </p>
            </div>
            <div className="flex gap-4">
              <button
                onClick={randomizeOutfit}
                className="btn-primary btn-lg flex items-center gap-3"
              >
                <Shuffle className="h-6 w-6" />
                <span>Randomize</span>
              </button>
              <button
                onClick={clearAll}
                className="btn-outline btn-lg"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Outfit Preview */}
          <div className="lg:col-span-1">
            <div className="outfit-card sticky top-24 animate-fade-in">
              <h2 className="heading-secondary text-foreground mb-8 text-center">Your Outfit</h2>

              {/* Outfit Display */}
              <div className="fashion-glow rounded-2xl p-10 mb-8 min-h-[400px] flex flex-col items-center justify-center gap-6">
                {selectedItems.top && (
                  <div className="text-7xl animate-scale-in">{selectedItems.top.icon}</div>
                )}
                {selectedItems.bottom && (
                  <div className="text-7xl animate-scale-in">{selectedItems.bottom.icon}</div>
                )}
                {selectedItems.shoes && (
                  <div className="text-6xl animate-scale-in">{selectedItems.shoes.icon}</div>
                )}
                {selectedItems.accessory && (
                  <div className="text-5xl animate-scale-in">{selectedItems.accessory.icon}</div>
                )}
                
                {!selectedItems.top && !selectedItems.bottom && !selectedItems.shoes && !selectedItems.accessory && (
                  <p className="text-muted text-center text-lg leading-relaxed">
                    Select items from your wardrobe to create an outfit
                  </p>
                )}
              </div>

              {/* Compatibility Score */}
              <div className={`p-8 rounded-2xl mb-8 border-2 transition-all ${
                compatibilityColor === 'green' ? 'bg-accent-50 dark:bg-accent-900/20 border-accent-300 dark:border-accent-700' :
                compatibilityColor === 'yellow' ? 'bg-secondary-50 dark:bg-secondary-900/20 border-secondary-300 dark:border-secondary-700' :
                compatibilityColor === 'red' ? 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700' :
                'bg-surface border-border'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <span className="font-semibold text-foreground text-lg">Compatibility Score</span>
                  {compatibility && (
                    <span className={`text-4xl font-bold ${
                      compatibilityColor === 'green' ? 'text-accent' :
                      compatibilityColor === 'yellow' ? 'text-secondary-600' :
                      'text-red-600'
                    }`}>
                      {compatibility}%
                    </span>
                  )}
                </div>
                
                {compatibility && (
                  <div className="h-3 bg-surface rounded-full overflow-hidden mb-4">
                    <div
                      className={`h-full rounded-full transition-all ${
                        compatibilityColor === 'green' ? 'bg-accent' :
                        compatibilityColor === 'yellow' ? 'bg-secondary-600' :
                        'bg-red-600'
                      }`}
                      style={{ width: `${compatibility}%` }}
                    ></div>
                  </div>
                )}

                <div className="flex items-start gap-2">
                  {compatibilityColor === 'green' ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  ) : compatibilityColor === 'yellow' ? (
                    <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  ) : compatibility ? (
                    <XCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <Info className="h-6 w-6 text-muted flex-shrink-0 mt-0.5" />
                  )}
                  <p className={`text-lg ${
                    compatibilityColor === 'green' ? 'text-accent dark:text-accent-300' :
                    compatibilityColor === 'yellow' ? 'text-secondary-600 dark:text-secondary-400' :
                    compatibility ? 'text-red-600 dark:text-red-400' : 'text-muted'
                  }`}>
                    {getCompatibilityMessage(compatibility)}
                  </p>
                </div>
              </div>

              {/* Detailed Scores */}
              {compatibility && (
                <div className="space-y-4 mb-8">
                  <h3 className="font-semibold text-foreground mb-4 text-lg">Breakdown</h3>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-muted">Color Harmony</span>
                    <div className="flex items-center gap-3">
                      <div className="w-28 h-3 bg-surface rounded-full overflow-hidden">
                        <div className="h-full bg-primary-600 rounded-full" style={{ width: '85%' }}></div>
                      </div>
                      <span className="font-medium text-foreground">85%</span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-muted">Style Match</span>
                    <div className="flex items-center gap-3">
                      <div className="w-28 h-3 bg-surface rounded-full overflow-hidden">
                        <div className="h-full bg-secondary-600 rounded-full" style={{ width: '78%' }}></div>
                      </div>
                      <span className="font-medium text-foreground">78%</span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-muted">Occasion Fit</span>
                    <div className="flex items-center gap-3">
                      <div className="w-28 h-3 bg-surface rounded-full overflow-hidden">
                        <div className="h-full bg-accent rounded-full" style={{ width: '90%' }}></div>
                      </div>
                      <span className="font-medium text-foreground">90%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-4">
                <button className="btn-primary btn-lg w-full flex items-center justify-center gap-3">
                  <Save className="h-6 w-6" />
                  <span>Save Outfit</span>
                </button>
                <button className="btn-outline btn-lg w-full flex items-center justify-center gap-3">
                  <Share2 className="h-6 w-6" />
                  <span>Share</span>
                </button>
              </div>
            </div>
          </div>

          {/* Wardrobe Selection */}
          <div className="lg:col-span-2 space-y-8 animate-fade-in">
            {/* Tops */}
            <div className="outfit-card">
              <h3 className="text-2xl font-bold text-foreground mb-6">Tops</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.tops.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('top', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.top?.id === item.id
                        ? 'border-primary-600 bg-primary-50 dark:bg-primary-900/20 shadow-lg'
                        : 'border-border hover:border-primary-300 hover:bg-surface-hover'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-foreground text-center truncate">
                      {item.name}
                    </p>
                    <div
                      className="w-6 h-6 rounded-full border-2 border-white shadow-md mx-auto mt-2"
                      style={{ backgroundColor: item.color }}
                    ></div>
                  </button>
                ))}
              </div>
            </div>

            {/* Bottoms */}
            <div className="outfit-card">
              <h3 className="text-2xl font-bold text-foreground mb-6">Bottoms</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.bottoms.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('bottom', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.bottom?.id === item.id
                        ? 'border-secondary-600 bg-secondary-50 dark:bg-secondary-900/20 shadow-lg'
                        : 'border-border hover:border-secondary-300 hover:bg-surface-hover'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-foreground text-center truncate">
                      {item.name}
                    </p>
                    <div
                      className="w-6 h-6 rounded-full border-2 border-white shadow-md mx-auto mt-2"
                      style={{ backgroundColor: item.color }}
                    ></div>
                  </button>
                ))}
              </div>
            </div>

            {/* Shoes */}
            <div className="outfit-card">
              <h3 className="text-2xl font-bold text-foreground mb-6">Shoes</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.shoes.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('shoes', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.shoes?.id === item.id
                        ? 'border-accent bg-accent-50 dark:bg-accent-900/20 shadow-lg'
                        : 'border-border hover:border-accent-300 hover:bg-surface-hover'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-foreground text-center truncate">
                      {item.name}
                    </p>
                    <div
                      className="w-6 h-6 rounded-full border-2 border-white shadow-md mx-auto mt-2"
                      style={{ backgroundColor: item.color }}
                    ></div>
                  </button>
                ))}
              </div>
            </div>

            {/* Accessories */}
            <div className="outfit-card">
              <h3 className="text-2xl font-bold text-foreground mb-6">Accessories</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.accessories.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('accessory', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.accessory?.id === item.id
                        ? 'border-primary-600 bg-primary-50 dark:bg-primary-900/20 shadow-lg'
                        : 'border-border hover:border-primary-300 hover:bg-surface-hover'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-foreground text-center truncate">
                      {item.name}
                    </p>
                    <div
                      className="w-6 h-6 rounded-full border-2 border-white shadow-md mx-auto mt-2"
                      style={{ backgroundColor: item.color }}
                    ></div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

