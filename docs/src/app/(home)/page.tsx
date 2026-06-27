import Link from 'next/link';

const FEATURES = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <path d="M12 2a4 4 0 0 1 4 4v1h3a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h3V6a4 4 0 0 1 4-4z"/>
        <path d="M9 8h6"/>
        <circle cx="12" cy="14" r="2"/>
      </svg>
    ),
    gradient: 'icon-bg-agent',
    title: 'Agent Loop',
    description:
      'The model thinks, calls your tools, sees results, and repeats. You describe the tools and the stopping rule — Aquaregia runs the loop.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 2a10 10 0 0 1 0 20"/>
        <path d="M2 12h20"/>
      </svg>
    ),
    gradient: 'icon-bg-provider',
    title: 'Multi-Provider',
    description:
      'Same code across OpenAI, Anthropic, Google, and any OpenAI-compatible endpoint. Swap the constructor — zero lock-in.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
      </svg>
    ),
    gradient: 'icon-bg-tools',
    title: 'Typed Tools',
    description:
      'A tool is a typed async fn. Derive JsonSchema, and Aquaregia builds the schema, validates the call, and marshals args into Rust.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M8 8h8"/>
        <path d="M8 12h8"/>
        <path d="M8 16h5"/>
      </svg>
    ),
    gradient: 'icon-bg-structured',
    title: 'Structured Output',
    description:
      'Call generate_object::&lt;T&gt;() and get a typed value back. JSON Schema derived automatically — your type, not a blob of text.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    ),
    gradient: 'icon-bg-streaming',
    title: 'Streaming',
    description:
      'Tokens arrive as they generate. Consume a uniform StreamEvent — text, reasoning, tool calls, usage — or pipe straight to SSE.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="size-5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="M9 12l2 2 4-4"/>
      </svg>
    ),
    gradient: 'icon-bg-production',
    title: 'Production Ready',
    description:
      'CancellationToken at every boundary. Exponential backoff with Retry-After. Typed ErrorCode — never match on strings.',
  },
];

const PROVIDERS = [
  'OpenAI',
  'Anthropic',
  'Google',
  'DeepSeek',
  'Together',
  'Groq',
  'Your gateway',
];

const CODE_LINES = [
  <>
    <span className="text-[#ff7b72]">use</span>{' '}
    <span className="text-[#d2a8ff]">aquaregia::providers::openai</span>;
  </>,
  '',
  <>
    <span className="text-[#8b949e]">#[</span><span className="text-[#d2a8ff]">tokio::main</span><span className="text-[#8b949e]">]</span>
  </>,
  <>
    <span className="text-[#ff7b72]">async fn</span>{' '}
    <span className="text-[#d2a8ff]">main</span>() <span className="text-[#ff7b72]">-&gt;</span>{' '}
    <span className="text-[#ffa657]">Result</span>&lt;(), <span className="text-[#ffa657]">Box</span>&lt;<span className="text-[#ff7b72]">dyn</span> <span className="text-[#ffa657]">Error</span>&gt;&gt; {'{'}
  </>,
  <>
    {'    '}<span className="text-[#ff7b72]">let</span>{' '}
    <span className="text-[#ffa657]">agent</span> <span className="text-[#ff7b72]">=</span>{' '}
    <span className="text-[#d2a8ff]">openai::Client::from_env</span>()?
  </>,
  <>
    {'        '}.<span className="text-[#d2a8ff]">agent</span>(<span className="text-[#a5d6ff]">&quot;gpt-4o-mini&quot;</span>)
  </>,
  <>
    {'        '}.<span className="text-[#d2a8ff]">build</span>()?;
  </>,
  '',
  <>
    {'    '}<span className="text-[#ff7b72]">let</span>{' '}
    <span className="text-[#ffa657]">response</span> <span className="text-[#ff7b72]">=</span>{' '}
    agent
  </>,
  <>
    {'        '}.<span className="text-[#d2a8ff]">prompt</span>(<span className="text-[#a5d6ff]">&quot;Explain Rust ownership in 3 bullet points.&quot;</span>)
  </>,
  <>
    {'        '}.<span className="text-[#ff7b72]">await</span>?;
  </>,
  '',
  <>
    {'    '}<span className="text-[#d2a8ff]">println!</span>(<span className="text-[#a5d6ff]">&quot;</span><span className="text-[#7ee787]">{'{response}'}</span><span className="text-[#a5d6ff]">&quot;</span>);
  </>,
  <>
    {'    '}<span className="text-[#d2a8ff]">Ok</span>(())
  </>,
  '}',
];

export default function HomePage() {
  return (
    <main>
      {/* ── Hero ── */}
      <section className="relative overflow-hidden bg-grid">
        {/* Glow orb behind code block */}
        <div className="glow-orb -top-32 left-1/2 size-[600px] -translate-x-1/2 bg-[#06b6d4]" />

        <div className="relative mx-auto max-w-5xl px-4 pb-20 pt-16 sm:px-6 sm:pt-24 sm:pb-28 lg:pt-32 lg:pb-36">
          <div className="text-center">
            <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
              Build LLM agents{' '}
              <span className="bg-gradient-to-r from-[#06b6d4] to-[#3b82f6] bg-clip-text text-transparent">
                in Rust
              </span>
            </h1>
            <p className="mx-auto mt-5 max-w-xl text-base text-fd-muted-foreground sm:text-lg">
              One API. Any provider. Aquaregia runs the agent loop —
              think, call tools, observe, repeat — so you don&apos;t write it yourself.
            </p>
            <div className="mt-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 font-mono text-sm backdrop-blur-sm">
                <span className="text-fd-muted-foreground">$</span>
                cargo add aquaregia
              </span>
            </div>
          </div>

          {/* Code block */}
          <div className="relative mx-auto mt-10 max-w-2xl sm:mt-12">
            {/* Outer glow */}
            <div className="absolute -inset-4 rounded-2xl bg-[#06b6d4]/5 blur-xl" />
            <div className="relative overflow-hidden rounded-xl border border-white/10 bg-[#0d1117]/90 shadow-2xl shadow-black/40 backdrop-blur-sm">
              {/* Title bar */}
              <div className="flex items-center border-b border-white/5 bg-white/[0.02] px-4 py-3">
                <span className="size-2.5 rounded-full bg-[#ff5f56]/80" />
                <span className="ml-1.5 size-2.5 rounded-full bg-[#ffbd2e]/80" />
                <span className="ml-1.5 size-2.5 rounded-full bg-[#27c93f]/80" />
                <span className="ml-4 font-mono text-xs text-white/30">main.rs</span>
              </div>
              {/* Code with line numbers */}
              <div className="flex">
                {/* Line numbers */}
                <div className="select-none border-r border-white/5 py-4 pl-4 pr-3 text-right font-mono text-xs leading-[1.75] text-white/15">
                  {CODE_LINES.map((_, i) => (
                    <div key={i}>{i > 0 ? i : ''}</div>
                  ))}
                </div>
                {/* Code content */}
                <pre className="overflow-x-auto px-4 py-4 text-sm leading-[1.75] sm:text-base">
                  <code className="font-mono text-[#c9d1d9]">
                    {CODE_LINES.map((line, i) => (
                      <div key={i}>{line || ' '}</div>
                    ))}
                  </code>
                </pre>
              </div>
            </div>
          </div>

          {/* CTAs */}
          <div className="mt-10 flex items-center justify-center gap-4">
            <Link
              href="/docs"
              className="group inline-flex items-center gap-2 rounded-lg bg-[#06b6d4] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[#06b6d4]/25 transition hover:bg-[#06b6d4]/90 hover:shadow-[#06b6d4]/40"
            >
              Get Started
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4 transition-transform group-hover:translate-x-0.5">
                <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
              </svg>
            </Link>
            <a
              href="https://github.com/minorcell/aquaregia"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-semibold backdrop-blur-sm transition hover:border-white/20 hover:bg-white/10"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="size-5 opacity-70">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="relative border-y border-white/5 bg-fd-background py-24 sm:py-32">
        <div className="mx-auto max-w-5xl px-4 sm:px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Everything you need to ship
            </h2>
            <p className="mt-4 text-fd-muted-foreground">
              From a single prompt to a multi-step agent loop — one dependency, any provider.
            </p>
          </div>
          <div className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="group relative rounded-xl border border-white/5 bg-white/[0.02] p-6 transition hover:border-white/10 hover:bg-white/[0.04]"
              >
                <div className={`mb-4 inline-flex size-10 items-center justify-center rounded-lg ${f.gradient} text-white shadow-lg`}>
                  {f.icon}
                </div>
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
      <section className="relative overflow-hidden py-24 sm:py-32">
        {/* Subtle glow top-right */}
        <div className="glow-orb -top-48 -right-48 size-[500px] bg-[#8b5cf6]" />
        <div className="relative mx-auto max-w-3xl px-4 text-center sm:px-6">
          <p className="font-mono text-xs font-medium tracking-widest text-fd-muted-foreground uppercase">
            Providers
          </p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            One API. Any model.
          </h2>
          <p className="mt-4 text-fd-muted-foreground">
            Same agent, same code — swap the constructor to change provider.
          </p>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-2.5">
            {PROVIDERS.map((p) => (
              <span
                key={p}
                className="inline-flex items-center rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium backdrop-blur-sm transition hover:border-white/20 hover:bg-white/[0.06]"
              >
                {p}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Bottom CTA ── */}
      <section className="border-t border-white/5 py-24 sm:py-32">
        <div className="mx-auto max-w-2xl px-4 text-center sm:px-6">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Ready to build?
          </h2>
          <p className="mt-4 text-fd-muted-foreground">
            One dependency. Any provider. Start building agents in minutes.
          </p>
          <div className="mt-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 font-mono text-sm backdrop-blur-sm">
              <span className="text-fd-muted-foreground">$</span>
              cargo add aquaregia
            </span>
          </div>
          <div className="mt-8">
            <Link
              href="/docs"
              className="group inline-flex items-center gap-2 rounded-lg bg-[#06b6d4] px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-[#06b6d4]/25 transition hover:bg-[#06b6d4]/90 hover:shadow-[#06b6d4]/40"
            >
              Read the Docs
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4 transition-transform group-hover:translate-x-0.5">
                <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
              </svg>
            </Link>
          </div>
          <p className="mt-12 text-xs text-fd-muted-foreground/60">
            MIT Licensed ·{' '}
            <a href="https://github.com/minorcell/aquaregia" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-fd-muted-foreground">GitHub</a>
            {' '}·{' '}
            <a href="https://crates.io/crates/aquaregia" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-fd-muted-foreground">crates.io</a>
            {' '}·{' '}
            <a href="https://docs.rs/aquaregia" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-fd-muted-foreground">docs.rs</a>
          </p>
        </div>
      </section>
    </main>
  );
}
