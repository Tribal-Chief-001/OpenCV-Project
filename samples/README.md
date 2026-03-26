# Sample Data

Place your test images and videos in this directory.

## Recommended Test Data

### Road Images (for lane + pothole detection)
You can use any road/highway images. Some free sources:

1. **Take your own photos** — Best option! Photograph roads around your campus/city
2. **Unsplash** (free, high-quality): Search "road", "highway", "pothole"
   - https://unsplash.com/s/photos/road
   - https://unsplash.com/s/photos/pothole
3. **Pexels** (free): https://www.pexels.com/search/road/

### Video (for motion analysis + tracking)
1. **Record dashcam footage** with your phone while commuting
2. **Pexels Videos** (free): https://www.pexels.com/search/videos/traffic/
3. **YouTube** (download with yt-dlp for educational use):
   - Search: "dashcam footage", "traffic video"

## Expected File Naming

```
samples/
├── road.jpg          # General road image
├── pothole.jpg       # Road with visible potholes
├── street.jpg        # Street scene with pedestrians/vehicles
├── highway.jpg       # Highway/lane image
└── traffic.mp4       # Traffic/driving video clip
```

## Notes
- Images: Any common format (jpg, png, bmp)
- Videos: MP4 recommended, AVI/MOV also supported
- Large video files are excluded from git (see .gitignore)
