import Link from 'next/link';

const FEATURES = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <path d="M12 2a4 4 0 0 1 4 4v1h3a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h3V6a4 4 0 0 1 4-4z"/>
        <path d="M9 8h6"/>
        <circle cx="12" cy="14" r="2"/>
      </svg>
    ),
    title: 'Agent Loop',
    description:
      'The model thinks, calls tools, sees results, and repeats until the job is done. You describe the tools and stopping rule; Aquaregia runs the loop.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 2a10 10 0 0 1 0 20"/>
        <path d="M2 12h20"/>
      </svg>
    ),
    title: 'Multi-Provider',
    description:
      'Use OpenAI, Anthropic, Google, or an OpenAI-compatible endpoint through provider-specific clients with the same high-level agent API.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
      </svg>
    ),
    title: 'Typed Tools',
    description:
      'A tool is a typed async function. Derive JsonSchema on the args, return any serializable value, and let Aquaregia handle schema and JSON plumbing.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M8 8h8"/>
        <path d="M8 12h8"/>
        <path d="M8 16h5"/>
      </svg>
    ),
    title: 'Structured Output',
    description:
      'Call generate_object::<T>() or stream_object::<T>() and receive typed Rust values with schemas derived from your own structs.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    ),
    title: 'Streaming',
    description:
      'Stream a single model call with StreamEvent, or stream a full agent run with AgentStreamEvent including model deltas, tools, steps, and final output.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-6">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="M9 12l2 2 4-4"/>
      </svg>
    ),
    title: 'Production Ready',
    description:
      'CancellationToken checked at every boundary. Exponential backoff with Retry-After honoured. Typed ErrorCode for control flow — never match on strings.',
  },
];

const PROVIDERS = [
  'OpenAI',
  'Anthropic',
  'Google',
  'OpenAI-compatible',
  'Local gateways',
  'Custom gateways',
];

export default function HomePage() {
  return (
    <main>
      {/* ── Hero ── */}
      <section className="relative overflow-hidden border-b bg-fd-card/50">
        <div className="mx-auto max-w-5xl px-4 pb-20 pt-16 sm:px-6 sm:pt-20 sm:pb-24 lg:pt-24 lg:pb-28">
          <div className="text-center">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
              Build LLM agents in Rust
            </h1>
            <p className="mx-auto mt-4 max-w-2xl text-lg text-fd-muted-foreground sm:text-xl">
              Aquaregia gives you the agent loop — think → call tools → observe → repeat —
              so you don&apos;t write it yourself. One API. Any provider.
            </p>
            <div className="mt-6">
              <code className="inline-block rounded-lg border bg-fd-secondary px-4 py-2 font-mono text-sm">
                $ cargo add aquaregia
              </code>
            </div>
          </div>

          {/* Code block */}
          <div className="mx-auto mt-8 max-w-2xl sm:mt-10">
            <div className="overflow-hidden rounded-xl border bg-[#0d1117] shadow-xl">
              <div className="flex items-center gap-1.5 border-b border-white/10 px-4 py-2.5">
                <span className="size-2.5 rounded-full bg-[#ff5f56]" />
                <span className="size-2.5 rounded-full bg-[#ffbd2e]" />
                <span className="size-2.5 rounded-full bg-[#27c93f]" />
                <span className="ml-3 font-mono text-xs text-white/50">main.rs</span>
              </div>
              <pre className="overflow-x-auto p-4 text-sm leading-relaxed sm:p-5 sm:text-base">
                <code className="font-mono text-[#c9d1d9]">
                  <span className="text-[#ff7b72]">use</span>{' '}
                  <span className="text-[#d2a8ff]">aquaregia::providers::openai</span>;<br />
                  <br />
                  <span className="text-[#8b949e]">#[</span><span className="text-[#d2a8ff]">tokio::main</span><span className="text-[#8b949e]">]</span><br />
                  <span className="text-[#ff7b72]">async fn</span>{' '}
                  <span className="text-[#d2a8ff]">main</span>() <span className="text-[#ff7b72]">-&gt;</span>{' '}
                  <span className="text-[#ffa657]">Result</span>&lt;(),{' '}
                  <span className="text-[#ffa657]">Box</span>&lt;<span className="text-[#ff7b72]">dyn</span>{' '}
                  <span className="text-[#ffa657]">std::error::Error</span>&gt;&gt; {'{'}<br />
                  {'    '}<span className="text-[#ff7b72]">let</span>{' '}
                  <span className="text-[#ffa657]">agent</span>{' '}
                  <span className="text-[#ff7b72]">=</span>{' '}
                  <span className="text-[#d2a8ff]">openai::Client::from_env</span>()?<br />
                  {'        '}.<span className="text-[#d2a8ff]">agent</span>(<span className="text-[#a5d6ff]">"gpt-5.5"</span>)<br />
                  {'        '}.<span className="text-[#d2a8ff]">build</span>()?;<br />
                  <br />
                  {'    '}<span className="text-[#ff7b72]">let</span>{' '}
                  <span className="text-[#ffa657]">response</span>{' '}
                  <span className="text-[#ff7b72]">=</span>{' '}
                  <span className="text-[#ffa657]">agent</span><br />
                  {'        '}.<span className="text-[#d2a8ff]">prompt</span>(<span className="text-[#a5d6ff]">"Explain Rust ownership in 3 bullet points."</span>)<br />
                  {'        '}.<span className="text-[#ff7b72]">await</span>?;<br />
                  <br />
                  {'    '}<span className="text-[#d2a8ff]">println!</span>(<span className="text-[#a5d6ff]">"</span><span className="text-[#7ee787]">{'{response}'}</span><span className="text-[#a5d6ff]">"</span>);<br />
                  {'    '}<span className="text-[#d2a8ff]">Ok</span>(())<br />
                  {'}'}
                </code>
              </pre>
            </div>
          </div>

          {/* CTAs */}
          <div className="mt-8 flex items-center justify-center gap-4 sm:mt-10">
            <Link
              href="/docs/quickstart"
              className="inline-flex items-center rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-semibold text-fd-primary-foreground shadow-sm transition hover:bg-fd-primary/90"
            >
              Get Started
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1.5 size-4">
                <path d="M5 12h14"/>
                <path d="m12 5 7 7-7 7"/>
              </svg>
            </Link>
            <a
              href="https://github.com/minorcell/aquaregia"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center rounded-lg border bg-fd-secondary px-5 py-2.5 text-sm font-semibold shadow-sm transition hover:bg-fd-secondary/80"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="mr-2 size-5">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="border-b py-20 sm:py-24">
        <div className="mx-auto max-w-5xl px-4 sm:px-6">
          <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl">
            What you can build
          </h2>
          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="rounded-xl border bg-fd-card p-6 transition-shadow hover:shadow-md"
              >
                <div className="mb-3 text-fd-primary">{f.icon}</div>
                <h3 className="font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-fd-muted-foreground">
                  {f.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Providers ── */}
      <section className="border-b py-20 sm:py-24">
        <div className="mx-auto max-w-3xl px-4 text-center sm:px-6">
          <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Runs on any provider
          </h2>
          <p className="mt-3 text-fd-muted-foreground">
            Same agent, same code. Swap the constructor to change provider.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            {PROVIDERS.map((p) => (
              <span
                key={p}
                className="inline-flex items-center rounded-full border bg-fd-secondary px-4 py-1.5 text-sm font-medium"
              >
                {p}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Bottom CTA ── */}
      <section className="py-20 sm:py-24">
        <div className="mx-auto max-w-3xl px-4 text-center sm:px-6">
          <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Ready to build?
          </h2>
          <p className="mt-3 text-fd-muted-foreground">
            One dependency. Any provider. Start building agents in minutes.
          </p>
          <div className="mt-6">
            <code className="inline-block rounded-lg border bg-fd-secondary px-4 py-2 font-mono text-sm">
              $ cargo add aquaregia
            </code>
          </div>
          <div className="mt-6">
            <Link
              href="/docs/quickstart"
              className="inline-flex items-center rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-semibold text-fd-primary-foreground shadow-sm transition hover:bg-fd-primary/90"
            >
              Read the Docs
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1.5 size-4">
                <path d="M5 12h14"/>
                <path d="m12 5 7 7-7 7"/>
              </svg>
            </Link>
          </div>
          <p className="mt-10 text-xs text-fd-muted-foreground">
            MIT Licensed ·{' '}
            <a
              href="https://github.com/minorcell/aquaregia"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              GitHub
            </a>
            {' '}·{' '}
            <a
              href="https://crates.io/crates/aquaregia"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              crates.io
            </a>
            {' '}·{' '}
            <a
              href="https://docs.rs/aquaregia"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              docs.rs
            </a>
          </p>
        </div>
      </section>
    </main>
  );
}
