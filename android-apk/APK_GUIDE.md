# Android APK Build & Download Guide · Caleb Yonli Face ID

This directory contains the complete native Android project template for the **Caleb Yonli Biometric Face ID** mobile application.

---

## ⚡ Method 1: Instant 1-Click Cloud APK via PWABuilder (Recommended - No Setup Needed)

Microsoft & Google provide **PWABuilder**, a free cloud service that packages your deployed PWA into a signed Android `.apk` in under 60 seconds:

### Steps:
1. Deploy your frontend to Vercel (or GitHub Pages).
2. Open **[https://www.pwabuilder.com](https://www.pwabuilder.com)** in your browser.
3. Paste your Vercel deployment URL (e.g. `https://your-caleb-faceid.vercel.app`) and click **Start**.
4. PWABuilder verifies your `manifest.webmanifest`, service worker `sw.js`, and high-res icons with a 100% PWA score.
5. Click **Package for Stores** -> Select **Android**.
6. Click **Generate / Download Package**.
7. Download the installable **`caleb-faceid.apk`**! Transfer it to any Android phone via WhatsApp, Telegram, or USB cable, tap to install, and run full-screen with native camera access.

---

## 📱 Method 2: Direct PWA Native Install (Zero Download, Instant App)

1. Open your deployed URL in Google Chrome or Microsoft Edge on your Android phone.
2. An automatic install banner will slide in: **"Install Caleb FaceID Mobile App on Home Screen"**.
3. Tap **Install** (or tap the 3-dots menu `⋮` -> **"Add to Home Screen"**).
4. The **Caleb FaceID** app icon appears on your home screen, opens full-screen like a native app without any browser URL bar, and supports offline loading!

---

## 💻 Method 3: Build Locally with Android Studio

If you have **Android Studio** installed:

1. Launch **Android Studio**.
2. Click **File** -> **Open** and select:
   `c:\Users\USER\Desktop\MY_PICTURES\android-apk`
3. Allow Gradle to sync.
4. Click **Build** -> **Build Bundle(s) / APK(s)** -> **Build APK(s)**.
5. In the bottom-right notification, click **locate** to find `app-debug.apk`.
6. Install on any connected Android device with `adb install app-debug.apk` or copy the file directly to your phone.

---

## 📦 Android Native Project Architecture

```
android-apk/
├── app/
│   ├── build.gradle                              # Android build configuration
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml               # Camera & Internet permissions
│           ├── java/com/caleb/faceid/
│           │   └── MainActivity.java             # Fullscreen WebView & WebChromeClient
│           ├── res/
│           │   ├── values/strings.xml            # App Title: Caleb FaceID
│           │   ├── values/styles.xml             # Cyberpunk dark theme
│           │   └── mipmap-xxhdpi/ic_launcher.png # High-res app launcher icon
│           └── assets/                           # Bundled mobile web app
│               ├── index.html
│               ├── style.css
│               ├── app.js
│               ├── sw.js
│               ├── manifest.webmanifest
│               └── icons/
├── build.gradle
├── settings.gradle
└── gradle.properties
```
