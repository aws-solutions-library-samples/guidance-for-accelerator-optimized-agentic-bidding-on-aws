import { gsap } from 'gsap';
import { ScrollToPlugin } from 'gsap/ScrollToPlugin';

// Register GSAP plugins used by the animation engine
gsap.registerPlugin(ScrollToPlugin);

export { gsap, ScrollToPlugin };
