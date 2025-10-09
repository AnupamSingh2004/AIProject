'use client';

import { useState } from 'react';
import { User, Mail, MapPin, Bell, Lock, Palette, Globe, Save, Camera, Edit2 } from 'lucide-react';

export default function ProfilePage() {
  const [activeTab, setActiveTab] = useState('profile');
  const [notifications, setNotifications] = useState({
    recommendations: true,
    wardrobe: true,
    updates: false,
  });

  const [profile, setProfile] = useState({
    name: 'Jane Doe',
    email: 'jane.doe@example.com',
    location: 'New York, USA',
    skinTone: 'Type IV (Medium-Dark)',
    undertone: 'Warm',
    stylePreferences: ['Casual', 'Business'],
  });

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'preferences', name: 'Style Preferences', icon: Palette },
    { id: 'settings', name: 'Settings', icon: Bell },
    { id: 'security', name: 'Security', icon: Lock },
  ];

  const styleCategories = ['Casual', 'Business', 'Formal', 'Athletic', 'Bohemian', 'Minimalist', 'Vintage', 'Trendy'];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">My Profile</h1>
          <p className="text-gray-600">Manage your account and preferences</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white border rounded-2xl shadow-md p-6 animate-fade-in">
              {/* Profile Picture */}
              <div className="text-center mb-6">
                <div className="relative inline-block">
                  <div className="w-24 h-24 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center text-white text-3xl font-bold mx-auto">
                    {profile.name.split(' ').map((n) => n[0]).join('')}
                  </div>
                  <button className="absolute bottom-0 right-0 p-2 bg-purple-600 text-white rounded-full shadow-lg hover:bg-purple-700 transition-colors">
                    <Camera className="h-4 w-4" />
                  </button>
                </div>
                <h2 className="text-xl font-bold text-gray-900 mt-4">{profile.name}</h2>
                <p className="text-gray-600 text-sm">{profile.email}</p>
              </div>

              {/* Navigation Tabs */}
              <nav className="space-y-2">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <tab.icon className="h-5 w-5" />
                    <span>{tab.name}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white border rounded-2xl shadow-md p-8 animate-fade-in">
              {/* Profile Tab */}
              {activeTab === 'profile' && (
                <div>
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold text-gray-900">Profile Information</h2>
                    <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                      <Edit2 className="h-4 w-4" />
                      <span>Edit</span>
                    </button>
                  </div>

                  <div className="space-y-6">
                    {/* Name */}
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Full Name
                      </label>
                      <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                        <User className="h-5 w-5 text-gray-400" />
                        <span className="text-gray-900">{profile.name}</span>
                      </div>
                    </div>

                    {/* Email */}
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Email Address
                      </label>
                      <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                        <Mail className="h-5 w-5 text-gray-400" />
                        <span className="text-gray-900">{profile.email}</span>
                      </div>
                    </div>

                    {/* Location */}
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-2">
                        Location
                      </label>
                      <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                        <MapPin className="h-5 w-5 text-gray-400" />
                        <span className="text-gray-900">{profile.location}</span>
                      </div>
                    </div>

                    {/* Skin Analysis Results */}
                    <div className="border-t border-gray-200 pt-6">
                      <h3 className="text-lg font-bold text-gray-900 mb-4">Skin Analysis</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-purple-50 rounded-lg">
                          <p className="text-sm text-gray-600 mb-1">Skin Type</p>
                          <p className="text-lg font-semibold text-purple-900">
                            {profile.skinTone}
                          </p>
                        </div>
                        <div className="p-4 bg-pink-50 rounded-lg">
                          <p className="text-sm text-gray-600 mb-1">Undertone</p>
                          <p className="text-lg font-semibold text-pink-900">
                            {profile.undertone}
                          </p>
                        </div>
                      </div>
                      <button className="mt-4 text-purple-600 font-medium hover:text-purple-700 transition-colors">
                        Re-analyze Skin Tone →
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Style Preferences Tab */}
              {activeTab === 'preferences' && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Style Preferences</h2>

                  <div className="space-y-6">
                    {/* Current Preferences */}
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-3">
                        Your Style Categories
                      </label>
                      <div className="flex flex-wrap gap-2 mb-4">
                        {profile.stylePreferences.map((style) => (
                          <span
                            key={style}
                            className="px-4 py-2 bg-purple-600 text-white rounded-full font-medium"
                          >
                            {style}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Available Categories */}
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-3">
                        Available Styles
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {styleCategories
                          .filter((style) => !profile.stylePreferences.includes(style))
                          .map((style) => (
                            <button
                              key={style}
                              className="px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-purple-100 hover:text-purple-700 transition-colors font-medium"
                            >
                              {style}
                            </button>
                          ))}
                      </div>
                    </div>

                    {/* Favorite Colors */}
                    <div className="border-t border-gray-200 pt-6">
                      <label className="block text-sm font-semibold text-gray-700 mb-3">
                        Favorite Colors
                      </label>
                      <div className="flex gap-3 mb-4">
                        {['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#000000'].map(
                          (color) => (
                            <div
                              key={color}
                              className="w-12 h-12 rounded-lg border-2 border-gray-300 cursor-pointer hover:border-purple-600 transition-colors"
                              style={{ backgroundColor: color }}
                            ></div>
                          )
                        )}
                      </div>
                    </div>

                    {/* Occasions */}
                    <div className="border-t border-gray-200 pt-6">
                      <label className="block text-sm font-semibold text-gray-700 mb-3">
                        Common Occasions
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {['Daily Wear', 'Work/Office', 'Parties', 'Gym', 'Formal Events', 'Dates'].map(
                          (occasion) => (
                            <label
                              key={occasion}
                              className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-purple-50 transition-colors"
                            >
                              <input type="checkbox" className="form-checkbox h-5 w-5 text-purple-600 rounded" />
                              <span className="text-gray-900">{occasion}</span>
                            </label>
                          )
                        )}
                      </div>
                    </div>

                    <button className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2">
                      <Save className="h-5 w-5" />
                      <span>Save Preferences</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Settings Tab */}
              {activeTab === 'settings' && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Settings</h2>

                  <div className="space-y-6">
                    {/* Notifications */}
                    <div>
                      <h3 className="text-lg font-bold text-gray-900 mb-4">Notifications</h3>
                      <div className="space-y-3">
                        <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer">
                          <div className="flex items-center gap-3">
                            <Bell className="h-5 w-5 text-gray-600" />
                            <div>
                              <p className="font-medium text-gray-900">Recommendation Updates</p>
                              <p className="text-sm text-gray-600">Get notified about new outfit suggestions</p>
                            </div>
                          </div>
                          <input
                            type="checkbox"
                            checked={notifications.recommendations}
                            onChange={(e) =>
                              setNotifications({ ...notifications, recommendations: e.target.checked })
                            }
                            className="form-checkbox h-6 w-6 text-purple-600 rounded"
                          />
                        </label>

                        <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer">
                          <div className="flex items-center gap-3">
                            <Bell className="h-5 w-5 text-gray-600" />
                            <div>
                              <p className="font-medium text-gray-900">Wardrobe Reminders</p>
                              <p className="text-sm text-gray-600">Reminders about unused items</p>
                            </div>
                          </div>
                          <input
                            type="checkbox"
                            checked={notifications.wardrobe}
                            onChange={(e) =>
                              setNotifications({ ...notifications, wardrobe: e.target.checked })
                            }
                            className="form-checkbox h-6 w-6 text-purple-600 rounded"
                          />
                        </label>

                        <label className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer">
                          <div className="flex items-center gap-3">
                            <Bell className="h-5 w-5 text-gray-600" />
                            <div>
                              <p className="font-medium text-gray-900">Feature Updates</p>
                              <p className="text-sm text-gray-600">News about new features</p>
                            </div>
                          </div>
                          <input
                            type="checkbox"
                            checked={notifications.updates}
                            onChange={(e) =>
                              setNotifications({ ...notifications, updates: e.target.checked })
                            }
                            className="form-checkbox h-6 w-6 text-purple-600 rounded"
                          />
                        </label>
                      </div>
                    </div>

                    {/* Language */}
                    <div className="border-t border-gray-200 pt-6">
                      <h3 className="text-lg font-bold text-gray-900 mb-4">Language</h3>
                      <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                        <Globe className="h-5 w-5 text-gray-400" />
                        <select className="flex-1 bg-transparent border-none focus:outline-none text-gray-900">
                          <option>English (US)</option>
                          <option>Spanish</option>
                          <option>French</option>
                          <option>German</option>
                          <option>Chinese</option>
                        </select>
                      </div>
                    </div>

                    <button className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2">
                      <Save className="h-5 w-5" />
                      <span>Save Settings</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Security Tab */}
              {activeTab === 'security' && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Security & Privacy</h2>

                  <div className="space-y-6">
                    {/* Change Password */}
                    <div>
                      <h3 className="text-lg font-bold text-gray-900 mb-4">Change Password</h3>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Current Password
                          </label>
                          <input
                            type="password"
                            className="w-full px-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                            placeholder="Enter current password"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            New Password
                          </label>
                          <input
                            type="password"
                            className="w-full px-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                            placeholder="Enter new password"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">
                            Confirm New Password
                          </label>
                          <input
                            type="password"
                            className="w-full px-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                            placeholder="Confirm new password"
                          />
                        </div>
                        <button className="w-full py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors">
                          Update Password
                        </button>
                      </div>
                    </div>

                    {/* Privacy */}
                    <div className="border-t border-gray-200 pt-6">
                      <h3 className="text-lg font-bold text-gray-900 mb-4">Privacy</h3>
                      <div className="space-y-4">
                        <button className="w-full text-left p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                          <p className="font-medium text-gray-900 mb-1">Download My Data</p>
                          <p className="text-sm text-gray-600">Get a copy of your data</p>
                        </button>
                        <button className="w-full text-left p-4 bg-red-50 rounded-lg hover:bg-red-100 transition-colors">
                          <p className="font-medium text-red-900 mb-1">Delete Account</p>
                          <p className="text-sm text-red-600">Permanently delete your account and data</p>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

