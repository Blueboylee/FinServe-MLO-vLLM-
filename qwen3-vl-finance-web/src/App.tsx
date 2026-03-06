import { FormEvent, useState } from "react";

type Role = "user" | "assistant" | "system";

type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type ApiMessage = { role: Role; content: string | ContentPart[] };

interface Message {
  id: string;
  role: Role;
  content: string;
  model?: string;
  imageDataUrl?: string;
  imageName?: string;
}

type ModelId = "./models/Qwen3-VL-8B-Instruct-AWQ-4bit" | "finance-expert-a" | "finance-expert-b";

const MODELS: { id: ModelId; label: string }[] = [
  { id: "./models/Qwen3-VL-8B-Instruct-AWQ-4bit", label: "基座模型" },
  { id: "finance-expert-a", label: "Expert A（宏观）" },
  { id: "finance-expert-b", label: "Expert B（行业/个股）" }
];

// Vite 开发时走 /api 代理到 vLLM；生产由 web_proxy 提供 /v1
const API_BASE = typeof import.meta !== "undefined" && (import.meta as any).env?.DEV ? "/api" : "";

async function chatRequestStream(
  model: string,
  messages: ApiMessage[],
  onChunk: (content: string) => void
): Promise<void> {
  const resp = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.7,
      max_tokens: 1024,
      stream: true
    })
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  const ct = (resp.headers.get("content-type") || "").toLowerCase();
  if (!ct.includes("text/event-stream")) {
    const text = await resp.text();
    throw new Error(text || `上游未返回流式数据（content-type=${ct}）`);
  }
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let full = "";
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") continue;
        try {
          const data = JSON.parse(raw);
          const delta = (data.choices?.[0]?.delta as { content?: string } | undefined)?.content;
          if (typeof delta === "string") {
            full += delta;
            onChunk(full);
          }
        } catch {
          /* ignore */
        }
      }
    }
  }
  if (buffer.startsWith("data: ")) {
    const raw = buffer.slice(6).trim();
    if (raw !== "[DONE]") {
      try {
        const data = JSON.parse(raw);
        const delta = (data.choices?.[0]?.delta as { content?: string } | undefined)?.content;
        if (typeof delta === "string") {
          full += delta;
          onChunk(full);
        }
      } catch {
        /* ignore */
      }
    }
  }
}

export default function App() {
  const [model, setModel] = useState<ModelId>("finance-expert-a");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);

  const toApiMessage = (m: Message): ApiMessage => {
    if (m.role === "user" && m.imageDataUrl) {
      const text = m.content.trim() || "请描述这张图片。";
      return {
        role: "user",
        content: [
          { type: "text", text },
          { type: "image_url", image_url: { url: m.imageDataUrl } }
        ]
      };
    }
    return { role: m.role, content: m.content };
  };

  const readFileAsDataUrl = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("读取图片失败"));
      reader.onload = () => resolve(String(reader.result));
      reader.readAsDataURL(file);
    });

  const normalizeToJpegDataUrl = async (file: File): Promise<{ dataUrl: string; name: string }> => {
    const src = await readFileAsDataUrl(file);
    const img = new Image();
    img.src = src;
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("解析图片失败（可能格式不支持）"));
    });

    const maxSide = 1024;
    const sw = img.naturalWidth || img.width;
    const sh = img.naturalHeight || img.height;
    const scale = Math.min(1, maxSide / Math.max(sw, sh));
    const w = Math.max(1, Math.round(sw * scale));
    const h = Math.max(1, Math.round(sh * scale));

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("无法创建 canvas");
    ctx.drawImage(img, 0, 0, w, h);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    const base = (file.name || "upload").replace(/\.[^.]+$/, "");
    return { dataUrl, name: `${base}.jpg` };
  };

  const onPickImage = async (file: File) => {
    if (file.size > 8 * 1024 * 1024) {
      setError("图片太大了（> 8MB），请换一张更小的。");
      return;
    }
    const out = await normalizeToJpegDataUrl(file);
    setImageDataUrl(out.dataUrl);
    setImageName(out.name);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !imageDataUrl) || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input,
      model,
      imageDataUrl: imageDataUrl ?? undefined,
      imageName: imageName ?? undefined
    };
    const botId = crypto.randomUUID();
    const botMsg: Message = {
      id: botId,
      role: "assistant",
      content: "",
      model
    };
    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput("");
    setImageDataUrl(null);
    setImageName(null);
    setError(null);
    setLoading(true);

    try {
      const historyForApi = [...messages, userMsg].map(toApiMessage);
      await chatRequestStream(model, historyForApi, (content) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === botId ? { ...m, content } : m))
        );
      });
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId ? { ...m, content: "请求失败，请重试。" } : m
        )
      );
      setError(err.message ?? String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Qwen3-VL Finance Chat</h1>
          <p>基于 vLLM，多专家（LoRA）金融助手</p>
        </div>
        <div className="model-select">
          <label>
            模型：
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as ModelId)}
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={handleClear}>
            清空对话
          </button>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`message message-${m.role}`}
            >
              <div className="message-meta">
                <span className="message-role">
                  {m.role === "user" ? "你" : "助手"}
                </span>
                {m.model && (
                  <span className="message-model">
                    {m.model === "finance-expert-a"
                      ? "Expert A"
                      : m.model === "finance-expert-b"
                      ? "Expert B"
                      : "基座"}
                  </span>
                )}
              </div>
              <div className="message-content">{m.content}</div>
              {m.imageDataUrl && (
                <img
                  className="message-image"
                  src={m.imageDataUrl}
                  alt={m.imageName ?? "uploaded"}
                />
              )}
            </div>
          ))}
          {messages.length === 0 && (
            <div className="empty-tip">
              选择一个模型，然后开始提问，例如：“请从宏观和行业两个角度分析 2024 年 A 股走势。”
            </div>
          )}
        </div>
      </main>

      <footer className="input-area">
        {error && <div className="error">请求出错：{error}</div>}
        <form onSubmit={handleSubmit}>
          <div className="input-attachments">
            <label className="file-input">
              图片：
              <input
                type="file"
                accept="image/*"
                disabled={loading}
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  try {
                    await onPickImage(file);
                  } catch (err: any) {
                    setError(err?.message ?? String(err));
                  } finally {
                    e.target.value = "";
                  }
                }}
              />
            </label>
            {imageDataUrl && (
              <div className="attach-preview">
                <img src={imageDataUrl} alt={imageName ?? "preview"} />
                <div className="attach-name">{imageName ?? "已选择图片"}</div>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setImageDataUrl(null);
                    setImageName(null);
                  }}
                >
                  移除
                </button>
              </div>
            )}
          </div>
          <textarea
            rows={3}
            placeholder="输入你的问题，按 Ctrl+Enter 发送..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                handleSubmit(e as any);
              }
            }}
          />
          <div className="input-actions">
            <span className="hint">
              当前模型：{
                MODELS.find((m) => m.id === model)?.label
              }
            </span>
            <button type="submit" disabled={loading}>
              {loading ? "生成中..." : "发送"}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}

