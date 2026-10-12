import ReactMarkdown from "react-markdown";

function normalizeAnswerMarkdown(answer: string) {
  let normalized = answer.trim();

  normalized = normalized.replace(/\r\n/g, "\n");
  normalized = normalized.replace(/:\s*\*\s*(\*\*[^\n]+)/g, ":\n\n- $1");
  normalized = normalized.replace(/\s+\*\*([^*]+)\*\*:/g, "\n- **$1:**");
  normalized = normalized.replace(/- \*\*([^*]+)\*\*:/g, "- **$1:**");
  normalized = normalized.replace(/\n{3,}/g, "\n\n");

  return normalized.trim();
}

export function StructuredMarkdown({ answer }: { answer: string }) {
  const content = normalizeAnswerMarkdown(answer);

  return (
    <div className="answer-markdown">
      <ReactMarkdown
        components={{
          h1: ({ children }) => <h1 className="answer-markdown__h1">{children}</h1>,
          h2: ({ children }) => <h2 className="answer-markdown__h2">{children}</h2>,
          h3: ({ children }) => <h3 className="answer-markdown__h3">{children}</h3>,
          p: ({ children }) => <p className="answer-markdown__p">{children}</p>,
          ul: ({ children }) => <ul className="answer-markdown__list">{children}</ul>,
          ol: ({ children }) => <ol className="answer-markdown__list answer-markdown__list--ordered">{children}</ol>,
          li: ({ children }) => <li className="answer-markdown__item">{children}</li>,
          strong: ({ children }) => <strong className="answer-markdown__strong">{children}</strong>,
          blockquote: ({ children }) => <blockquote className="answer-markdown__quote">{children}</blockquote>,
          code: ({ children }) => <code className="answer-markdown__code">{children}</code>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
