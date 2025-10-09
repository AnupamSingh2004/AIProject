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
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Mix & Match</h1>
              <p className="text-gray-600">
                Create and test outfit combinations from your wardrobe
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={randomizeOutfit}
                className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-full font-semibold hover:bg-purple-700 transition-colors shadow-lg"
              >
                <Shuffle className="h-5 w-5" />
                <span>Randomize</span>
              </button>
              <button
                onClick={clearAll}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-full font-semibold hover:bg-gray-300 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Outfit Preview */}
          <div className="lg:col-span-1">
            <div className="bg-white border rounded-2xl shadow-lg p-8 sticky top-24 animate-fade-in">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Your Outfit</h2>

              {/* Outfit Display */}
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-8 mb-6 min-h-[300px] flex flex-col items-center justify-center gap-4">
                {selectedItems.top && (
                  <div className="text-6xl">{selectedItems.top.icon}</div>
                )}
                {selectedItems.bottom && (
                  <div className="text-6xl">{selectedItems.bottom.icon}</div>
                )}
                {selectedItems.shoes && (
                  <div className="text-5xl">{selectedItems.shoes.icon}</div>
                )}
                {selectedItems.accessory && (
                  <div className="text-4xl">{selectedItems.accessory.icon}</div>
                )}
                
                {!selectedItems.top && !selectedItems.bottom && !selectedItems.shoes && !selectedItems.accessory && (
                  <p className="text-gray-400 text-center">
                    Select items from your wardrobe to create an outfit
                  </p>
                )}
              </div>

              {/* Compatibility Score */}
              <div className={`p-6 rounded-xl mb-6 ${
                compatibilityColor === 'green' ? 'bg-green-50 border-2 border-green-200' :
                compatibilityColor === 'yellow' ? 'bg-yellow-50 border-2 border-yellow-200' :
                compatibilityColor === 'red' ? 'bg-red-50 border-2 border-red-200' :
                'bg-gray-50 border-2 border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-3">
                  <span className="font-semibold text-gray-900">Compatibility Score</span>
                  {compatibility && (
                    <span className={`text-3xl font-bold ${
                      compatibilityColor === 'green' ? 'text-green-600' :
                      compatibilityColor === 'yellow' ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {compatibility}%
                    </span>
                  )}
                </div>
                
                {compatibility && (
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden mb-3">
                    <div
                      className={`h-full rounded-full transition-all ${
                        compatibilityColor === 'green' ? 'bg-green-500' :
                        compatibilityColor === 'yellow' ? 'bg-yellow-500' :
                        'bg-red-500'
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
                    <Info className="h-5 w-5 text-gray-600 flex-shrink-0 mt-0.5" />
                  )}
                  <p className={`text-sm ${
                    compatibilityColor === 'green' ? 'text-green-800' :
                    compatibilityColor === 'yellow' ? 'text-yellow-800' :
                    compatibility ? 'text-red-800' : 'text-gray-700'
                  }`}>
                    {getCompatibilityMessage(compatibility)}
                  </p>
                </div>
              </div>

              {/* Detailed Scores */}
              {compatibility && (
                <div className="space-y-3 mb-6">
                  <h3 className="font-semibold text-gray-900 mb-3">Breakdown</h3>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Color Harmony</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div className="h-full bg-purple-600 rounded-full" style={{ width: '85%' }}></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">85%</span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Style Match</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div className="h-full bg-pink-600 rounded-full" style={{ width: '78%' }}></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">78%</span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Occasion Fit</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-600 rounded-full" style={{ width: '90%' }}></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">90%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-3">
                <button className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2">
                  <Save className="h-5 w-5" />
                  <span>Save Outfit</span>
                </button>
                <button className="w-full py-3 bg-gray-100 text-gray-700 rounded-xl font-medium hover:bg-gray-200 transition-colors flex items-center justify-center gap-2">
                  <Share2 className="h-5 w-5" />
                  <span>Share</span>
                </button>
              </div>
            </div>
          </div>

          {/* Wardrobe Selection */}
          <div className="lg:col-span-2 space-y-6 animate-fade-in">
            {/* Tops */}
            <div className="bg-white border rounded-2xl shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Tops</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.tops.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('top', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.top?.id === item.id
                        ? 'border-purple-600 bg-purple-50 shadow-lg'
                        : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-gray-900 text-center truncate">
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
            <div className="bg-white border rounded-2xl shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Bottoms</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.bottoms.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('bottom', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.bottom?.id === item.id
                        ? 'border-purple-600 bg-purple-50 shadow-lg'
                        : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-gray-900 text-center truncate">
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
            <div className="bg-white border rounded-2xl shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Shoes</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.shoes.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('shoes', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.shoes?.id === item.id
                        ? 'border-purple-600 bg-purple-50 shadow-lg'
                        : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-gray-900 text-center truncate">
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
            <div className="bg-white border rounded-2xl shadow-md p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Accessories</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {wardrobeItems.accessories.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectItem('accessory', item)}
                    className={`p-4 rounded-xl border-2 transition-all ${
                      selectedItems.accessory?.id === item.id
                        ? 'border-purple-600 bg-purple-50 shadow-lg'
                        : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="text-4xl mb-2 text-center">{item.icon}</div>
                    <p className="text-sm font-medium text-gray-900 text-center truncate">
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

