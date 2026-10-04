// Receives the list of dropped object IDs and forwards it to Telegram.
// Secrets (set with `wrangler secret put`): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Optional: ACCESS_CODE.
const ID_RE = /^[RM]\d{3}$/;
const json = (o, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/api/submit") return env.ASSETS.fetch(request);
    if (request.method !== "POST") return json({ error: "POST only" }, 405);

    let body;
    try { body = await request.json(); } catch { return json({ error: "bad json" }, 400); }
    if (env.ACCESS_CODE && body.code !== env.ACCESS_CODE) return json({ error: "wrong access code" }, 403);

    const dropped = [...new Set(Array.isArray(body.dropped) ? body.dropped : [])];
    if (dropped.length > 200 || !dropped.every((i) => typeof i === "string" && ID_RE.test(i)))
      return json({ error: "invalid ids" }, 400);
    dropped.sort();
    const r = dropped.filter((i) => i[0] === "R"), m = dropped.filter((i) => i[0] === "M");
    const text =
      `Drop list submitted\n` +
      `Rickshaw: drop ${r.length}, keep ${100 - r.length}\n${r.join(", ") || "-"}\n\n` +
      `Motorcycle: drop ${m.length}, keep ${100 - m.length}\n${m.join(", ") || "-"}`;

    const res = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text }),
    });
    if (!res.ok) return json({ error: "telegram failed" }, 502);
    return json({ ok: true, dropped: dropped.length });
  },
};
