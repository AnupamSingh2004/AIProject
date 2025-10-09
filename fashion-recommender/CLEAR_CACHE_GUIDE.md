# Quick Reference - How to Clear Browser Cache

## 🌐 Chrome / Edge / Brave

### Method 1: DevTools (Fastest)
1. Press `F12` or `Ctrl + Shift + I`
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"

### Method 2: Application Tab
1. Press `F12`
2. Click **Application** tab
3. Click **Clear storage** in left sidebar
4. Click "Clear site data" button
5. Hard refresh: `Ctrl + Shift + R`

### Method 3: Settings
1. Press `Ctrl + Shift + Delete`
2. Select "All time"
3. Check:
   - ☑️ Cookies and other site data
   - ☑️ Cached images and files
4. Click "Clear data"

---

## 🦊 Firefox

### Method 1: DevTools
1. Press `F12`
2. Click **Storage** tab
3. Right-click **Local Storage**
4. Select "Delete All"
5. Hard refresh: `Ctrl + Shift + R`

### Method 2: Settings
1. Press `Ctrl + Shift + Delete`
2. Select "Everything"
3. Check:
   - ☑️ Cookies
   - ☑️ Cache
4. Click "Clear Now"

---

## 🍎 Safari (Mac)

### Method 1: Developer Tools
1. Enable Developer menu: Safari → Preferences → Advanced → Show Develop menu
2. Develop → Empty Caches
3. Hard refresh: `Cmd + Shift + R`

### Method 2: Complete Clear
1. Safari → Preferences → Privacy
2. Click "Manage Website Data"
3. Click "Remove All"
4. Click "Done"

---

## 🔍 Console Method (All Browsers)

### Universal Command
1. Press `F12` (DevTools)
2. Go to **Console** tab
3. Paste and press Enter:

```javascript
// Clear all storage
localStorage.clear();
sessionStorage.clear();

// Reload page
location.reload(true);
```

---

## ✅ After Clearing

You should see:
- ✅ Light gray background
- ✅ No theme toggle button
- ✅ Clean white cards
- ✅ Purple accent colors
- ✅ No console errors

---

## ⚠️ If Still Seeing Dark Mode

Try **Incognito/Private Mode**:
- Chrome: `Ctrl + Shift + N`
- Firefox: `Ctrl + Shift + P`
- Safari: `Cmd + Shift + N`

Then visit: http://localhost:3000

If it works in incognito, your regular browser has stubborn cache - try clearing again or restart browser completely.

---

## 🔄 Quick Test

Run in Console to check current state:
```javascript
// Should show light colors
console.log('Background:', getComputedStyle(document.body).backgroundColor);

// Should NOT exist
console.log('Theme in storage:', localStorage.getItem('theme'));

// Should be false  
console.log('Has dark class:', document.documentElement.classList.contains('dark'));
```

Expected output:
```
Background: rgb(249, 250, 251)  // Light gray
Theme in storage: null           // No theme stored
Has dark class: false            // No dark class
```

---

Happy browsing with your clean light theme! ✨
