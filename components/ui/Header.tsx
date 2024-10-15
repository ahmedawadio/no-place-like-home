import { Search } from "lucide-react";

export function Header() {
  return (
    <header className="absolute top-0 z-100 flex items-center justify-between w-full h-16 px-5 bg-gradient-to-b from-background/10 via-background/50 to-background/80 backdrop-blur-xl">
      <a
        href="https://www.ahmedawad.io/"
        className="flex items-center cursor-pointer opacity-80"
      >
        <span className="text-xl sm:text-2xl">Nest ML</span>
      </a>

      <a
          href="/"
          className="flex items-center cursor-pointer opacity-60 text-sm sm:text-base"
        >
      <div className="flex items-center border-2 border-opacity-40 border-white rounded-3xl py-2 px-4 ">
          <Search size={18} className="mr-2" />
          New Search
      </div>
        </a>
    </header>
  );
}
