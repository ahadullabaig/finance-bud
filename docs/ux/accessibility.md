# Accessibility Checklist (WCAG & UX Guidance)

This document provides a compact checklist and recommended automated tools for verifying accessibility across the UI.

## Checklist (target WCAG AA)

- [ ] All interactive elements have visible focus states
- [ ] All images and meaningful icons include `alt` text or an ARIA label
- [ ] Keyboard navigation: all flows reachable via keyboard only
- [ ] Color contrast: meet WCAG AA contrast ratios for text and UI components
- [ ] Form controls: labels associated with inputs; error states announced
- [ ] Semantic HTML used for structure (headings, lists, landmarks)
- [ ] Screen-reader testing for main flows (plan creation, ReasonGraph exploration)
- [ ] Resizable text and responsive layouts for small viewports

## Automated tools

- axe (Deque) — use `npm install -g axe-core` or use the browser extension
- pa11y — `npm i -g pa11y`
- Lighthouse — built into Chrome DevTools or CLI via `npm i -g lighthouse`
- Playwright / Puppeteer — use for headless interaction + axe-runner for automated checks

## Quick example (Playwright + axe)

1. Install Playwright: `npm i -D @playwright/test`
2. Add a small test that loads the page and runs axe-core to collect violations

## Notes

- Prioritize main user flows for accessibility fixes first (create plan, view ReasonGraph, replay demo).
- Document keyboard shortcuts and provide an accessibility statement in the site footer.
