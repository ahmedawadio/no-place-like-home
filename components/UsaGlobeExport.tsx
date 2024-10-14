import dynamic from 'next/dynamic';


const UsaGlobe = dynamic(() => import("./UsaGlobe"), { ssr: false });
export default UsaGlobe;
