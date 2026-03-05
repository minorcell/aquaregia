import { notFound } from "next/navigation";
import { LOCALES, isValidLocale } from "../../lib/site-config";

export function generateStaticParams() {
  return LOCALES.map((lang) => ({ lang }));
}

export default async function LocaleLayout({ children, params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }
  return children;
}
