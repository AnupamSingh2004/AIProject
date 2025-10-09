# StyleAI - AI-Powered Fashion Recommendation System

A modern, responsive Next.js application that provides personalized fashion recommendations based on skin tone analysis and style preferences.

## 🌟 Features

### 1. **Skin Tone Analysis**
- Upload photos for AI-powered skin tone detection
- Fitzpatrick scale classification (Type I-VI)
- Undertone determination (warm/cool/neutral)
- Personalized color recommendations
- Confidence scoring

### 2. **Smart Wardrobe Management**
- Digital wardrobe with visual gallery
- Upload and categorize clothing items
- Filter by category, season, and color
- Grid and list view options
- Search functionality
- Edit and delete items

### 3. **AI Recommendations**
- Personalized outfit suggestions
- Filter by occasion (Casual, Business, Formal, etc.)
- Filter by season (Spring, Summer, Fall, Winter)
- Compatibility scores for each outfit
- Detailed explanations for recommendations
- Color palette visualization

### 4. **Mix & Match Interface**
- Interactive outfit builder
- Real-time compatibility scoring
- Color harmony analysis
- Style matching feedback
- Save and share outfits

### 5. **User Profile & Settings**
- Profile management
- Style preference customization
- Notification settings
- Dark mode support
- Multi-language support (placeholder)
- Privacy and security settings

## 🚀 Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn package manager

### Installation

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Run the development server**
   ```bash
   npm run dev
   ```

3. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
fashion-recommender/
├── app/
│   ├── analyze/          # Skin tone analysis page
│   ├── wardrobe/         # Wardrobe management page
│   ├── recommendations/  # Recommendations dashboard
│   ├── mix-match/        # Mix & match interface
│   ├── profile/          # User profile and settings
│   ├── layout.tsx        # Root layout with header/footer
│   ├── page.tsx          # Homepage
│   └── globals.css       # Global styles
├── components/
│   ├── Header.tsx        # Navigation header
│   └── Footer.tsx        # Site footer
├── public/               # Static assets
└── package.json          # Dependencies and scripts
```

## 🎨 Technology Stack

- **Framework**: Next.js 15.5.4 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **Runtime**: React 19.1.0

## 📱 Responsive Design

The application is fully responsive and optimized for:
- 📱 Mobile devices (320px+)
- 📱 Tablets (768px+)
- 💻 Desktops (1024px+)
- 🖥️ Large screens (1440px+)

## 🎯 Current Features Status

### ✅ Completed
- Landing page with feature showcase
- Responsive header and footer
- Skin tone analysis UI (placeholder for AI integration)
- Wardrobe management with filters
- Recommendations dashboard
- Mix & match outfit builder
- User profile and settings
- Mobile-friendly navigation

### 🔄 Ready for Backend Integration
- User authentication
- Photo upload and processing
- AI skin tone detection
- Clothing item recognition
- Recommendation algorithm
- Data persistence
- User wardrobe storage

## 🔮 Future Enhancements

### Phase 1: AI Integration
- [ ] Integrate skin tone detection ML model
- [ ] Implement clothing recognition model
- [ ] Deploy recommendation algorithm
- [ ] Set up image storage (AWS S3/Cloud Storage)

### Phase 2: Backend & Database
- [ ] User authentication (JWT/OAuth2)
- [ ] RESTful API development
- [ ] Database schema implementation (PostgreSQL/MongoDB)
- [ ] User wardrobe persistence
- [ ] Outfit history tracking

### Phase 3: Advanced Features
- [ ] Color theory-based recommendations
- [ ] Seasonal color analysis
- [ ] Virtual try-on (AR)
- [ ] Social features (sharing, following)
- [ ] Style trends integration

## 🎨 Color Palette

- **Primary**: Purple (#8b5cf6)
- **Secondary**: Pink (#ec4899)
- **Accent**: Amber (#f59e0b)
- **Background**: White (#ffffff) / Dark (#0a0a0a)

## 🛠️ Development Commands

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start
```

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

Built with ❤️ by the StyleAI Team

---

**Note**: This is the frontend MVP. All UI features are fully functional with mock data. The application is ready for AI model and backend integration as specified in the requirements document.
