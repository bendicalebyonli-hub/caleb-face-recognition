const CACHE_NAME = 'caleb-face-id-v2.1.0';
const ASSETS_TO_CACHE = [
  '/',
  '/style.css',
  '/app.js',
  '/manifest.webmanifest',
  '/icons/favicon.svg',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/apple-touch-icon.png',
  '/icons/icon-maskable.png'
];

// Install Event
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      for (const asset of ASSETS_TO_CACHE) {
        try {
          await cache.add(asset);
        } catch (err) {
          console.warn('SW cache.add skipped asset:', asset, err);
        }
      }
    }).then(() => self.skipWaiting())
  );
});

// Activate Event - clean up stale caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event - network first for API, cache first for app shell
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // If request is to backend API, bypass service worker cache
  if (url.pathname.includes('/recognize') || 
      url.pathname.includes('/status') || 
      url.pathname.includes('/enroll') ||
      url.hostname.includes('onrender.com') ||
      url.hostname.includes('railway.app') ||
      url.port === '5000') {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        fetch(event.request).then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, networkResponse);
            });
          }
        }).catch(() => {});
        return cachedResponse;
      }
      return fetch(event.request).catch(() => caches.match('/'));
    })
  );
});
