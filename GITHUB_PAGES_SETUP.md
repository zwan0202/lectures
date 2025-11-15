# GitHub Pages Setup Instructions

This document explains how to deploy the CS336 Executable Lectures to GitHub Pages.

## Overview

The repository is configured to automatically build and deploy to GitHub Pages whenever changes are pushed to the `main` branch. The workflow:

1. Executes Python lecture files to generate trace JSON files
2. Builds the React frontend with Vite
3. Deploys everything to GitHub Pages

## Prerequisites

Before the automated deployment will work, you need to configure GitHub Pages in your repository settings.

## Setup Steps

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (top navigation)
3. Scroll down to **Pages** (left sidebar under "Code and automation")
4. Under **Source**, select:
   - **Source**: GitHub Actions (instead of "Deploy from a branch")

### 2. Verify Repository Settings

Make sure your repository has:
- Actions enabled (Settings → Actions → General)
- Workflow permissions set to "Read and write permissions" (Settings → Actions → General → Workflow permissions)

### 3. Trigger the Deployment

The workflow will automatically run when you:
- Push to the `main` branch
- Manually trigger it from the Actions tab

To manually trigger:
1. Go to **Actions** tab in your repository
2. Select "Build and Deploy to GitHub Pages" workflow
3. Click "Run workflow" button
4. Select the `main` branch
5. Click "Run workflow"

### 4. Check Deployment Status

1. Go to **Actions** tab
2. Click on the running workflow
3. Monitor the progress of both the "build" and "deploy" jobs
4. Once complete, your site will be available at:
   ```
   https://<username>.github.io/<repository-name>/
   ```
   For example: `https://zwan0202.github.io/lectures/`

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy-github-pages.yml   # GitHub Actions workflow
├── images/                            # Static images used in lectures
├── trace-viewer/                      # React frontend application
│   ├── src/                           # React source code
│   ├── public/                        # Static assets (populated during build)
│   ├── package.json
│   └── vite.config.js                # Base path: /spring2025-lectures/
├── lecture_*.py                       # Executable lecture files
├── execute.py                         # Trace generator
├── requirements.txt                   # Python dependencies
└── var/traces/                        # Generated trace files (gitignored)
```

## How It Works

### Build Process

1. **Python Execution Phase**:
   - Sets up Python 3.11 environment
   - Installs dependencies from `requirements.txt`
   - Executes lecture files: `lecture_01`, `lecture_02`, `lecture_06`, etc.
   - Generates trace JSON files in `var/traces/`

2. **Frontend Build Phase**:
   - Sets up Node.js 20 environment
   - Copies generated traces and images to `trace-viewer/public/`
   - Builds React app with Vite (configured for `/spring2025-lectures/` base path)
   - Outputs to `trace-viewer/dist/`

3. **Deployment Phase**:
   - Uploads build artifacts to GitHub Pages
   - Deploys to your GitHub Pages URL

### URL Structure

Once deployed, lectures can be accessed at:
```
https://<username>.github.io/<repository-name>/?trace=var/traces/lecture_01.json
```

The base path `/spring2025-lectures/` is configured in:
- `trace-viewer/vite.config.js`
- `trace-viewer/src/App.jsx`

**Note**: If your repository name is different from `spring2025-lectures`, you'll need to update these base paths.

## Customization

### Changing the Base Path

If you need to change the repository name or base path:

1. Update `trace-viewer/vite.config.js`:
   ```javascript
   base: process.env.NODE_ENV === 'production' ? '/your-repo-name/' : '/',
   ```

2. Update `trace-viewer/src/App.jsx`:
   ```javascript
   <BrowserRouter basename={process.env.NODE_ENV === 'production' ? '/your-repo-name/' : '/'}>
   ```

3. Commit and push the changes

### Adding More Lectures

To add more lectures to the build:

1. Edit `.github/workflows/deploy-github-pages.yml`
2. Find the "Generate lecture traces" step
3. Add your new lecture module to the command:
   ```yaml
   python execute.py -m lecture_01 lecture_02 ... lecture_NEW
   ```

## Troubleshooting

### Workflow Fails on Python Dependencies

Some dependencies (like `triton`, `kenlm`, `fasttext`) may require system libraries or may not be available on all platforms. The workflow uses `continue-on-error: true` to allow the build to proceed even if some lectures fail to execute.

If you need all lectures to succeed:
1. Remove `continue-on-error: true` from the workflow
2. Adjust `requirements.txt` to exclude problematic dependencies
3. Or add system dependencies to the workflow

### Site Not Loading / 404 Errors

1. Check that GitHub Pages is configured to use "GitHub Actions" as source
2. Verify the base path matches your repository name
3. Check the Actions tab for any failed workflows
4. Ensure the deployment job completed successfully

### Lectures Not Displaying

1. Verify trace JSON files were generated (check workflow logs)
2. Check browser console for errors
3. Ensure images are being copied correctly
4. Verify the trace URL parameter is correct

## Local Development

To test locally before deploying:

```bash
# Generate trace files
python execute.py -m lecture_01

# Copy assets to public
rm -f trace-viewer/public/images trace-viewer/public/var
cp -r images trace-viewer/public/
cp -r var trace-viewer/public/

# Run development server
cd trace-viewer
npm install
npm run dev

# Visit: http://localhost:5173?trace=var/traces/lecture_01.json
```

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Vite Documentation](https://vitejs.dev/)
