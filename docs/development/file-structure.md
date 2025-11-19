# Finance-Bud Project File Structure

## Overview

This document describes the organized file structure of the **finance-bud** project after reorganization. The project uses a modular, layered architecture to separate concerns and maintain scalability.

---

## Directory Structure

```
finance-bud/
├── App.tsx                          # Main app entry point (root component)
├── main.tsx                         # React app initialization (imports App + styles)
├── index.html                       # HTML entry point (Vite)
├── vite.config.ts                   # Vite build configuration
├── package.json                     # Dependencies & npm scripts
├── README.md                        # Project README
├── LICENSE                          # License file
│
├── components/                      # Reusable UI layer
│   └── ui/                          # 48+ Radix UI primitives + custom components
│       ├── button.tsx               # Button component (variants: default, outline, ghost, etc.)
│       ├── card.tsx                 # Card container component
│       ├── tabs.tsx                 # Tabbed interface
│       ├── input.tsx                # Text input field
│       ├── textarea.tsx             # Textarea field
│       ├── select.tsx               # Select dropdown
│       ├── dialog.tsx               # Modal dialog
│       ├── dropdown-menu.tsx        # Dropdown menu
│       ├── tooltip.tsx              # Tooltip
│       ├── popover.tsx              # Popover
│       ├── form.tsx                 # Form utilities
│       ├── table.tsx                # Data table
│       ├── pagination.tsx           # Pagination
│       ├── slider.tsx               # Range slider
│       ├── switch.tsx               # Toggle switch
│       ├── checkbox.tsx             # Checkbox
│       ├── radio-group.tsx          # Radio group
│       ├── progress.tsx             # Progress bar
│       ├── skeleton.tsx             # Loading skeleton
│       ├── avatar.tsx               # Avatar image
│       ├── badge.tsx                # Badge label
│       ├── breadcrumb.tsx           # Breadcrumb nav
│       ├── sidebar.tsx              # Sidebar layout
│       ├── sheet.tsx                # Sheet panel
│       ├── drawer.tsx               # Drawer panel
│       ├── scroll-area.tsx          # Scrollable area
│       ├── resizable.tsx            # Resizable panels
│       ├── accordion.tsx            # Accordion component
│       ├── alert.tsx                # Alert message
│       ├── alert-dialog.tsx         # Alert dialog
│       ├── command.tsx              # Command palette
│       ├── context-menu.tsx         # Context menu
│       ├── carousel.tsx             # Image carousel
│       ├── calendar.tsx             # Date calendar
│       ├── chart.tsx                # Chart wrapper
│       ├── collapsible.tsx          # Collapsible panel
│       ├── hover-card.tsx           # Hover card
│       ├── input-otp.tsx            # OTP input
│       ├── label.tsx                # Form label
│       ├── menubar.tsx              # Menu bar
│       ├── navigation-menu.tsx      # Navigation menu
│       ├── separator.tsx            # Horizontal separator
│       ├── sonner.tsx               # Toast notifications
│       ├── toggle.tsx               # Toggle button
│       ├── toggle-group.tsx         # Toggle group
│       └── ImageWithFallback.tsx    # Image with fallback
│
├── views/                           # Feature/page components (screen-level)
│   ├── ArchitectureView.tsx         # System architecture visualization
│   ├── DashboardView.tsx            # Metrics/dashboard view
│   ├── LiveDemoView.tsx             # Interactive live demo
│   ├── ReasonGraphView.tsx          # Reason graph visualization
│   ├── ReasonGraphLive.tsx          # Live reason graph processing
│   ├── CMVLTriggerPanel.tsx         # CMVL trigger controls
│   └── FinancialPlanComparison.tsx  # Financial plan comparison
│
├── hooks/                           # Custom React hooks
│   └── use-mobile.ts                # Hook to detect mobile viewport
│
├── lib/                             # Shared utilities & helpers
│   └── utils.ts                     # Utility functions (e.g., `cn()` classname merger)
│
├── styles/                          # Global stylesheets & documentation
│   ├── index.css                    # Main CSS stylesheet
│   ├── globals.css                  # Global CSS utilities
│   ├── Guidelines.md                # Design guidelines
│   └── Attributions.md              # Attribution & credits
│
├── docs/                            # Project documentation
│   └── FILE_STRUCTURE.md            # This file
│
├── .gitignore                       # Git ignore rules
└── build/                           # (Generated) Production bundle
    ├── index.html
    ├── assets/
    │   ├── index-*.js               # Minified JavaScript
    │   └── index-*.css              # Minified CSS
    └── ...
```

---

## Architecture Layers

### 1. **Entry Layer** (Root)
- `index.html` — Static HTML entry point
- `main.tsx` — React initialization and global styles import
- `App.tsx` — Root React component, defines main UI structure

### 2. **Components Layer** (`components/ui/`)
- **Purpose:** Reusable, atomic UI components (buttons, cards, inputs, etc.)
- **Technology:** Built with Radix UI primitives + Tailwind CSS
- **Usage:** Imported by views and higher-level components
- **Styling:** All components use utility-first CSS via Tailwind
- **Pattern:** Each file exports one or more related component functions

### 3. **Views Layer** (`views/`)
- **Purpose:** Page-level and feature-level components
- **Responsibility:** Compose UI components into complete features/screens
- **Examples:**
  - `ArchitectureView` — Multi-agent system architecture diagram
  - `DashboardView` — Key metrics display
  - `LiveDemoView` — Interactive demo with state management
  - `CMVLTriggerPanel` — Trigger controls for market/life events
- **Pattern:** Each view imports components from `components/ui/` and renders complete feature UI

### 4. **Utilities Layer** (`lib/`, `hooks/`)
- **`lib/utils.ts`** — Shared functions (class name merging via `cn()`)
- **`hooks/use-mobile.ts`** — Responsive design detection hook
- **Pattern:** Single-responsibility, reusable across layers

### 5. **Styles Layer** (`styles/`)
- **`index.css`** — Main application stylesheet
- **`globals.css`** — Global CSS utilities and resets
- **`.md` files** — Design documentation and guidelines

---

## Key Import Patterns

### From Root Component
```tsx
// App.tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { ArchitectureView } from './views/ArchitectureView';
import { DashboardView } from './views/DashboardView';
import { LiveDemoView } from './views/LiveDemoView';
```

### From View Components
```tsx
// views/ArchitectureView.tsx
import { Card } from '../components/ui/card';
import { motion } from 'motion/react';
```

### From UI Components
```tsx
// components/ui/button.tsx
import { cn } from '../../lib/utils';
```

---

## Build & Development

### Scripts
- `npm run dev` — Start Vite dev server (hot reload)
- `npm run build` — Build production bundle to `build/` folder

### Build Configuration
- **Bundler:** Vite (fast, modern)
- **Framework:** React 18.3.1
- **Styling:** Tailwind CSS
- **UI Library:** Radix UI
- **Icons:** Lucide React
- **Animations:** Framer Motion
- **Forms:** React Hook Form
- **Charts:** Recharts
- **Tables:** Custom + shadcn/ui patterns

---

## File Organization Best Practices

### Adding New UI Components
1. Create a new file in `components/ui/ComponentName.tsx`
2. Import shared utilities: `import { cn } from '../../lib/utils';`
3. Use Radix UI primitives as base when possible
4. Export component(s) at the end of the file
5. Import in views or other components as needed

### Adding New Views/Pages
1. Create a new file in `views/ViewName.tsx`
2. Import UI components: `import { Card, Button } from '../components/ui/...';`
3. Compose the view using UI components
4. Export the view component
5. Import and register in `App.tsx` tab structure if needed

### Adding Utilities
1. Create a new file in `lib/util-name.ts` or `hooks/use-name.ts`
2. Export named functions/hooks
3. Import in components where needed

---

## Folder Rationale

| Folder | Purpose | Imports From |
|--------|---------|--------------|
| `components/ui/` | Atomic UI primitives | `lib/`, external packages |
| `views/` | Feature-level components | `components/ui/`, `lib/`, `hooks/` |
| `lib/` | Shared utilities | Nothing (standalone) |
| `hooks/` | Custom React hooks | External packages |
| `styles/` | Global CSS & docs | (CSS files, no imports) |
| `docs/` | Project documentation | (Markdown, no code) |

---

## Summary

This organized structure enables:
- **Scalability:** Easy to add new components and views without cluttering the root
- **Maintainability:** Clear separation of concerns (UI primitives vs. features vs. utilities)
- **Reusability:** Components in `components/ui/` are used across multiple views
- **Clarity:** New developers can quickly understand where to find/add code
- **Performance:** Tree-shaking and code-splitting via Vite for optimized builds

For questions or updates to this structure, refer to the main `README.md`.
