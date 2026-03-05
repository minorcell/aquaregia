import Link from "next/link";

export default function NotFound() {
  return (
    <main className="aq-home">
      <section className="aq-hero">
        <h1>404</h1>
        <p>The page you requested does not exist.</p>
        <div className="aq-actions">
          <Link className="aq-btn aq-btn-primary" href="/zh">
            Go to Home
          </Link>
          <Link className="aq-btn" href="/zh/docs">
            Browse Docs
          </Link>
        </div>
      </section>
    </main>
  );
}
