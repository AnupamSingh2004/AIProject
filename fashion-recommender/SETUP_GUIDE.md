# Fashion Recommender - Setup & Run Guide

## 🚀 How to Run the Development Server

### ✅ Correct Way (from fashion-recommender directory)

```bash
# Navigate to the fashion-recommender directory
cd fashion-recommender

# Run the development server
npm run dev
```

**OR** from the root AIProject directory:

```bash
cd C:\Users\Prachi\Desktop\qq\AIProject\fashion-recommender
npm run dev
```

### ❌ Common Mistake

Running `npm run dev` from the root `AIProject` directory will cause this error:
```
npm error enoent Could not read package.json
```

**Why?** The `package.json` file is located in the `fashion-recommender` subdirectory, not in the root `AIProject` directory.

---

## 🎨 Theme Toggle Fix

### Problem That Was Fixed
The theme toggle wasn't working because the `ThemeProvider` was conditionally rendering its context provider based on the `mounted` state. This caused components to try to access the theme context before it was available.

### Solution Applied
- Removed the conditional rendering in `ThemeProvider.tsx`
- The context provider now always wraps the children, ensuring the theme context is always available
- This allows all pages and components to access the theme state immediately

---

## 🌐 Accessing the Application

Once the dev server is running, you can access the application at:

- **Local:** http://localhost:3000
- **Network:** http://172.27.34.28:3000 (accessible from other devices on your network)

---

## 🔧 Troubleshooting

### If you see "useTheme must be used within a ThemeProvider" error:
1. Clear the Next.js cache: `Remove-Item -Recurse -Force .next`
2. Restart the dev server: `npm run dev`
3. Hard refresh your browser: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)

### If the theme toggle doesn't work:
1. Open browser DevTools (F12)
2. Check the Console for any errors
3. Verify localStorage is enabled in your browser
4. Clear browser cache and reload

### If styles aren't updating:
1. Make sure Turbopack is running (it's the default in Next.js 15)
2. Check that you're editing files in the correct directory
3. Save the file to trigger hot reload

---

## 📁 Project Structure

```
AIProject/
├── fashion-recommender/          ← Run npm commands HERE
│   ├── app/                      ← Next.js pages
│   │   ├── layout.tsx           ← Root layout with ThemeProvider
│   │   ├── page.tsx             ← Home page
│   │   ├── analyze/             ← Skin tone analysis page
│   │   ├── wardrobe/            ← Wardrobe management
│   │   ├── recommendations/     ← AI recommendations
│   │   ├── mix-match/           ← Outfit mixing
│   │   └── profile/             ← User profile
│   ├── components/              ← React components
│   │   ├── ThemeProvider.tsx   ← Theme context provider
│   │   ├── ClientHeader.tsx    ← Header with theme toggle
│   │   └── Footer.tsx          ← Footer component
│   ├── public/                  ← Static assets
│   ├── package.json            ← Project dependencies
│   └── next.config.ts          ← Next.js configuration
└── (other backend files)
```

---

## 🎯 Quick Commands

```powershell
# Navigate to project
cd C:\Users\Prachi\Desktop\qq\AIProject\fashion-recommender

# Install dependencies (if needed)
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run production build
npm start

# Clear Next.js cache
Remove-Item -Recurse -Force .next
```

---

## ✨ Features Working

- ✅ Theme toggle (light/dark mode)
- ✅ Persistent theme preference (saved to localStorage)
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Smooth theme transitions
- ✅ All pages properly themed
- ✅ Header and profile theme controls synchronized

---

## 🐛 Known Issues & Solutions

### Issue: "Cannot read package.json"
**Solution:** Make sure you're in the `fashion-recommender` directory

### Issue: Theme doesn't persist after refresh
**Solution:** Check that localStorage is not disabled in your browser

### Issue: Hot reload not working
**Solution:** Save the file again or restart the dev server

---

## 📝 Making Changes

### To modify styling:
- Edit `globals.css` for global styles
- Use Tailwind classes with `dark:` prefix for dark mode variants
- All components automatically respond to theme changes

### To add new pages:
- Create a new folder in `app/` directory
- Add a `page.tsx` file inside
- The route is automatically created based on folder name

### To modify the theme:
- Edit `components/ThemeProvider.tsx` for theme logic
- Edit CSS variables in `app/globals.css` for colors
- All pages will automatically update

---

## 🎨 Theme Colors Reference

### Light Mode
- Background: `#ffffff` / `bg-gray-50`
- Text: `#1f2937` / `text-gray-900`
- Primary: `#8b5cf6` (Purple)
- Secondary: `#ec4899` (Pink)

### Dark Mode
- Background: `#1a1d3a` / `dark:bg-[#1a1d3a]`
- Text: `#e5e7eb` / `dark:text-gray-100`
- Primary: `#a78bfa` (Light Purple)
- Secondary: `#f472b6` (Light Pink)

---

Happy coding! 🚀
