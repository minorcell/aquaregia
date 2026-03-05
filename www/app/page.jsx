import { redirect } from "next/navigation";
import { DEFAULT_LOCALE } from "../lib/site-config";

export default function RootRedirectPage() {
  redirect(`/${DEFAULT_LOCALE}`);
}
