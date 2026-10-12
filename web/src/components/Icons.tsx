import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function BaseIcon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    />
  );
}

export function LibraryIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M3 6.5A2.5 2.5 0 0 1 5.5 4H20v15.5A1.5 1.5 0 0 0 18.5 18H5.5A2.5 2.5 0 0 0 3 20.5z" />
      <path d="M7 8h8" />
      <path d="M7 12h10" />
    </BaseIcon>
  );
}

export function AskIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M9.4 9.3a2.9 2.9 0 1 1 4.4 2.5c-.9.5-1.3 1-1.3 1.9" />
      <circle cx="12" cy="16.9" r=".7" fill="currentColor" stroke="none" />
    </BaseIcon>
  );
}

export function ChatIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M6 17.5 3.5 20V7a3 3 0 0 1 3-3h11a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H6z" />
      <path d="M8 10h8" />
      <path d="M8 13h5" />
    </BaseIcon>
  );
}

export function EvalIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M5 18h14" />
      <path d="M7 18V9" />
      <path d="M12 18V6" />
      <path d="M17 18v-4" />
    </BaseIcon>
  );
}

export function UploadIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M12 16V5" />
      <path d="m7.5 9.5 4.5-4.5 4.5 4.5" />
      <path d="M4 18.5h16" />
    </BaseIcon>
  );
}

export function ThemeIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M12 3.5a8.5 8.5 0 1 0 8.5 8.5A6.5 6.5 0 0 1 12 3.5Z" />
    </BaseIcon>
  );
}

export function SettingsIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <circle cx="12" cy="12" r="2.5" />
      <path d="M12 4.5v1.7" />
      <path d="M12 17.8v1.7" />
      <path d="m5.9 5.9 1.2 1.2" />
      <path d="m16.9 16.9 1.2 1.2" />
      <path d="M4.5 12h1.7" />
      <path d="M17.8 12h1.7" />
      <path d="m5.9 18.1 1.2-1.2" />
      <path d="m16.9 7.1 1.2-1.2" />
    </BaseIcon>
  );
}

export function SignOutIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M10 5H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h3" />
      <path d="M13 16l4-4-4-4" />
      <path d="M17 12H9" />
    </BaseIcon>
  );
}

export function MenuIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M4 7h16" />
      <path d="M4 12h16" />
      <path d="M4 17h16" />
    </BaseIcon>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M6 6l12 12" />
      <path d="M18 6 6 18" />
    </BaseIcon>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M20 5v5h-5" />
      <path d="M4 19v-5h5" />
      <path d="M6.8 9A7 7 0 0 1 18 7.4L20 10" />
      <path d="M17.2 15A7 7 0 0 1 6 16.6L4 14" />
    </BaseIcon>
  );
}

export function ResetIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M5 6.5h5" />
      <path d="M9 6.5V4.5" />
      <path d="M7 10v6" />
      <path d="M12 10v6" />
      <path d="M17 10v6" />
      <path d="M5 6.5h14" />
      <path d="M6 6.5l1 12h10l1-12" />
    </BaseIcon>
  );
}

export function FolderIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M3.5 7.5A2.5 2.5 0 0 1 6 5h4l2 2h6A2.5 2.5 0 0 1 20.5 9.5v7A2.5 2.5 0 0 1 18 19H6a2.5 2.5 0 0 1-2.5-2.5z" />
    </BaseIcon>
  );
}

export function FileIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="M8 3.5h6l4.5 4.5V19a1.5 1.5 0 0 1-1.5 1.5H8A2.5 2.5 0 0 1 5.5 18V6A2.5 2.5 0 0 1 8 3.5z" />
      <path d="M14 3.5V8h4.5" />
      <path d="M9 12h6" />
      <path d="M9 15.5h4.5" />
    </BaseIcon>
  );
}

export function ChevronIcon(props: IconProps) {
  return (
    <BaseIcon {...props}>
      <path d="m9 6 6 6-6 6" />
    </BaseIcon>
  );
}
