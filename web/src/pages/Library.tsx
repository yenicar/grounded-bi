import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods as ForceGraph2DMethods } from "react-force-graph-2d";
import * as THREE from "three";

import {
  api,
  type CorpusHealthResponse,
  type CorpusRecord,
  type DocumentListResponse,
  type DocumentRecord,
} from "../api/client";
import {
  ChevronIcon,
  CloseIcon,
  FileIcon,
  FolderIcon,
  LibraryIcon,
  RefreshIcon,
  ResetIcon,
  UploadIcon,
} from "../components/Icons";
import { LibrarySkeleton } from "../components/LoadingSkeleton";
import { ProgressCard } from "../components/ProgressCard";
import { RadialActionMenu } from "../components/RadialActionMenu";

type ExplorerFileNode = {
  id: string;
  type: "file";
  label: string;
  relativePath: string;
  document: DocumentRecord;
};

type ExplorerFolderNode = {
  id: string;
  type: "folder";
  label: string;
  path: string;
  folders: ExplorerFolderNode[];
  files: ExplorerFileNode[];
};

type KnowledgeGraphNodeKind = "corpus" | "folder" | "document" | "category" | "state";

type KnowledgeGraphNode = {
  id: string;
  kind: KnowledgeGraphNodeKind;
  label: string;
  detail: string;
  count?: number;
  size: number;
  color: string;
  x: number;
  y: number;
  documentId?: string;
  folderId?: string;
};

type KnowledgeGraphLink = {
  id: string;
  from: string;
  to: string;
  tone: "primary" | "category" | "state";
};

type KnowledgeGraph = {
  nodes: KnowledgeGraphNode[];
  links: KnowledgeGraphLink[];
};

function formatDocDate(value?: string | null): string | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function createFolderNode(id: string, label: string, path: string): ExplorerFolderNode {
  return {
    id,
    type: "folder",
    label,
    path,
    folders: [],
    files: [],
  };
}

function sortFolderTree(node: ExplorerFolderNode) {
  node.folders.sort((a, b) => a.label.localeCompare(b.label));
  node.files.sort((a, b) => a.label.localeCompare(b.label));
  node.folders.forEach(sortFolderTree);
}

function buildExplorerTree(documents: DocumentRecord[]): ExplorerFolderNode {
  const root = createFolderNode("root", "HR Resources", "");

  for (const document of documents) {
    const relativePath = document.source_relative_path ?? document.doc_name;
    const parts = relativePath.split(/[\\/]/).filter(Boolean);
    const fileLabel = parts.pop() ?? document.doc_name;

    let current = root;
    const lineage: string[] = [];

    for (const segment of parts) {
      lineage.push(segment);
      const currentPath = lineage.join("/");
      let folder = current.folders.find((item) => item.path === currentPath);
      if (!folder) {
        folder = createFolderNode(`folder:${currentPath}`, segment, currentPath);
        current.folders.push(folder);
      }
      current = folder;
    }

    current.files.push({
      id: `file:${relativePath}`,
      type: "file",
      label: fileLabel,
      relativePath,
      document,
    });
  }

  sortFolderTree(root);
  return root;
}

function collectFolderIds(node: ExplorerFolderNode): string[] {
  return [node.id, ...node.folders.flatMap(collectFolderIds)];
}

function findSelectedDocument(root: ExplorerFolderNode, selectedId: string | null): ExplorerFileNode | null {
  for (const file of root.files) {
    if (file.id === selectedId) {
      return file;
    }
  }
  for (const folder of root.folders) {
    const result = findSelectedDocument(folder, selectedId);
    if (result) {
      return result;
    }
  }
  return null;
}

function findFirstFile(root: ExplorerFolderNode): ExplorerFileNode | null {
  if (root.files[0]) {
    return root.files[0];
  }
  for (const folder of root.folders) {
    const nested = findFirstFile(folder);
    if (nested) {
      return nested;
    }
  }
  return null;
}

function collectDocuments(node: ExplorerFolderNode): number {
  return node.files.length + node.folders.reduce((total, child) => total + collectDocuments(child), 0);
}

function collectFiles(node: ExplorerFolderNode): ExplorerFileNode[] {
  return [...node.files, ...node.folders.flatMap(collectFiles)];
}

function slugifyGraphId(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "unknown";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function hashToUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function extractGraphTerms(document: DocumentRecord): string[] {
  const text = `${document.doc_name} ${document.doc_summary ?? ""} ${document.doc_type ?? ""}`;
  const stopWords = new Set([
    "and",
    "for",
    "from",
    "with",
    "the",
    "this",
    "that",
    "policy",
    "policies",
    "document",
    "resources",
    "employee",
    "employees",
  ]);
  return [...new Set(text.match(/[A-Za-z][A-Za-z-]{3,}/g) ?? [])]
    .map((term) => term.toLowerCase())
    .filter((term) => !stopWords.has(term))
    .slice(0, 4);
}

function buildKnowledgeGraph({
  activeCorpus,
  documents,
  explorerTree,
}: {
  activeCorpus: CorpusRecord | null;
  documents: DocumentRecord[];
  explorerTree: ExplorerFolderNode;
}): KnowledgeGraph {
  const palette = ["#df5b64", "#71bf67", "#6f95d8", "#d7923a", "#c286bd", "#d8c943", "#72c4c7", "#a7a9af"];
  const nodes: KnowledgeGraphNode[] = [];
  const links: KnowledgeGraphLink[] = [];

  function addNode(node: KnowledgeGraphNode) {
    if (!nodes.some((existing) => existing.id === node.id)) {
      nodes.push(node);
    }
  }

  addNode({
    id: "corpus:active",
    kind: "corpus",
    label: activeCorpus?.name ?? "HR Resources",
    detail: `${documents.length} files / ${activeCorpus?.chunk_count ?? 0} chunks`,
    count: documents.length,
    size: 16,
    color: "#79c8c9",
    x: 50,
    y: 50,
  });

  const topFolders = explorerTree.folders.length ? explorerTree.folders : [explorerTree];
  const allFiles = collectFiles(explorerTree);
  const categoryNames = [...new Set(documents.map((document) => document.policy_category).filter(Boolean) as string[])];
  const clusterNames = categoryNames.length >= 2 ? categoryNames : topFolders.map((folder) => folder.label);
  const clusterRadius = clusterNames.length > 7 ? 32 : 28;

  clusterNames.forEach((cluster, index) => {
    const angle = -Math.PI / 2 + (index / Math.max(1, clusterNames.length)) * Math.PI * 2;
    const clusterId = `category:${slugifyGraphId(cluster)}`;
    const clusterDocs = documents.filter((document) => document.policy_category === cluster);
    addNode({
      id: clusterId,
      kind: "category",
      label: cluster,
      detail: `${clusterDocs.length || collectDocuments(topFolders[index] ?? explorerTree)} related files`,
      count: clusterDocs.length,
      size: 11,
      color: palette[index % palette.length],
      x: clamp(50 + Math.cos(angle) * clusterRadius, 13, 87),
      y: clamp(50 + Math.sin(angle) * clusterRadius, 12, 88),
    });
    links.push({ id: `link:corpus:${clusterId}`, from: "corpus:active", to: clusterId, tone: "category" });
  });

  topFolders.forEach((folder, index) => {
    const angle = -Math.PI / 2 + (index / Math.max(1, topFolders.length)) * Math.PI * 2;
    const folderId = folder.id === "root" ? "folder:workspace" : folder.id;
    const folderCount = collectDocuments(folder);
    addNode({
      id: folderId,
      kind: "folder",
      label: folder.label,
      detail: `${folderCount} documents`,
      count: folderCount,
      size: 9,
      color: palette[index % palette.length],
      folderId: folder.id,
      x: clamp(50 + Math.cos(angle) * 18, 18, 82),
      y: clamp(50 + Math.sin(angle) * 18, 18, 82),
    });
    links.push({ id: `link:corpus:${folderId}`, from: "corpus:active", to: folderId, tone: "primary" });
  });

  allFiles.slice(0, 140).forEach((file, index) => {
    const document = file.document;
    const categoryIndex = Math.max(0, clusterNames.findIndex((cluster) => cluster === document.policy_category));
    const fallbackIndex = index % Math.max(1, clusterNames.length);
    const clusterIndex = categoryIndex >= 0 ? categoryIndex : fallbackIndex;
    const clusterAngle = -Math.PI / 2 + (clusterIndex / Math.max(1, clusterNames.length)) * Math.PI * 2;
    const orbit = 12 + hashToUnit(file.relativePath) * 18;
    const localAngle = clusterAngle + (hashToUnit(`${file.relativePath}:angle`) - 0.5) * 1.7;
    const documentId = `document:${file.relativePath}`;
    const color = palette[clusterIndex % palette.length];
    const folderRoot = file.relativePath.split(/[\\/]/).filter(Boolean)[0];
    const folderId = folderRoot ? `folder:${folderRoot}` : "folder:workspace";
    const categoryId = document.policy_category ? `category:${slugifyGraphId(document.policy_category)}` : null;

    addNode({
      id: documentId,
      kind: "document",
      label: document.doc_name,
      detail: document.policy_category ?? document.source_level ?? document.doc_format.toUpperCase(),
      documentId: file.id,
      size: 4.5 + Math.min(5, document.table_count + (document.page_count ? 1 : 0)),
      color,
      x: clamp(50 + Math.cos(localAngle) * orbit + (hashToUnit(`${file.relativePath}:x`) - 0.5) * 8, 5, 95),
      y: clamp(50 + Math.sin(localAngle) * orbit + (hashToUnit(`${file.relativePath}:y`) - 0.5) * 8, 5, 95),
    });
    links.push({ id: `link:${folderId}:${documentId}`, from: folderId, to: documentId, tone: "primary" });
    if (categoryId) {
      links.push({ id: `link:${documentId}:${categoryId}`, from: documentId, to: categoryId, tone: "category" });
    }

    extractGraphTerms(document).forEach((term, termIndex) => {
      const termId = `term:${slugifyGraphId(term)}`;
      const termAngle = localAngle + (termIndex - 1.5) * 0.38;
      addNode({
        id: termId,
        kind: "state",
        label: term,
        detail: "Concept term",
        size: 3.8,
        color,
        x: clamp(50 + Math.cos(termAngle) * (orbit + 8 + termIndex * 2), 4, 96),
        y: clamp(50 + Math.sin(termAngle) * (orbit + 8 + termIndex * 2), 4, 96),
      });
      links.push({ id: `link:${documentId}:${termId}`, from: documentId, to: termId, tone: "state" });
    });
  });

  const warningCount = documents.filter((document) => document.warnings.length > 0).length;
  const driftCount = documents.filter((document) => document.drift_status && document.drift_status !== "current").length;
  [
    { id: "state:warnings", label: "Warnings", count: warningCount, x: 10, y: 10 },
    { id: "state:drift", label: "Drift", count: driftCount, x: 90, y: 10 },
  ].forEach((state) => {
    nodes.push({
      id: state.id,
      kind: "state",
      label: state.label,
      detail: `${state.count} files need attention`,
      count: state.count,
      size: 8,
      color: "#e06b72",
      x: state.x,
      y: state.y,
    });
    links.push({ id: `link:corpus:${state.id}`, from: "corpus:active", to: state.id, tone: "state" });
  });

  return { nodes, links };
}

function renderTree({
  node,
  depth,
  expandedFolderIds,
  onToggleFolder,
  selectedDocumentId,
  onSelectDocument,
}: {
  node: ExplorerFolderNode;
  depth: number;
  expandedFolderIds: Set<string>;
  onToggleFolder: (id: string) => void;
  selectedDocumentId: string | null;
  onSelectDocument: (id: string) => void;
}): JSX.Element {
  const isExpanded = expandedFolderIds.has(node.id);

  return (
    <div key={node.id} className="library-drawer__tree-branch">
      <button
        type="button"
        className={`library-drawer__tree-row library-drawer__tree-row--folder ${isExpanded ? "library-drawer__tree-row--expanded" : ""}`}
        style={{ paddingLeft: `${0.8 + depth * 1.05}rem` }}
        onClick={() => onToggleFolder(node.id)}
      >
        <span className={`library-drawer__tree-caret ${isExpanded ? "library-drawer__tree-caret--open" : ""}`}>
          <ChevronIcon width={12} height={12} />
        </span>
        <FolderIcon width={15} height={15} />
        <span className="library-drawer__tree-label">{node.label}</span>
        <small>{collectDocuments(node)}</small>
      </button>

      {isExpanded ? (
        <div className="library-drawer__tree-children">
          {node.folders.map((folder) =>
            renderTree({
              node: folder,
              depth: depth + 1,
              expandedFolderIds,
              onToggleFolder,
              selectedDocumentId,
              onSelectDocument,
            }),
          )}
          {node.files.map((file) => (
            <button
              key={file.id}
              type="button"
              className={`library-drawer__tree-row library-drawer__tree-row--file ${selectedDocumentId === file.id ? "library-drawer__tree-row--active" : ""}`}
              style={{ paddingLeft: `${1.25 + (depth + 1) * 1.05}rem` }}
              onClick={() => onSelectDocument(file.id)}
            >
              <span className="library-drawer__tree-caret library-drawer__tree-caret--placeholder" />
              <FileIcon width={14} height={14} />
              <span className="library-drawer__tree-label">{file.label}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// ─── Force-directed graph data shapes ───────────────────────────────────────
type FGNode = KnowledgeGraphNode & { x?: number; y?: number; z?: number; vx?: number; vy?: number; vz?: number; fx?: number; fy?: number; fz?: number };
type FGLink = { id: string; source: string; target: string; tone: KnowledgeGraphLink["tone"] };
type FGData = { nodes: FGNode[]; links: FGLink[] };

function ThreeKnowledgeScene({
  data,
  width,
  height,
  activeNodeId,
  activeLinkNodeIds,
  onNodeClick,
  getNodeColor,
  getLinkColor,
  getLinkWidth,
}: {
  data: FGData;
  width: number;
  height: number;
  activeNodeId: string;
  activeLinkNodeIds: Set<string>;
  onNodeClick: (node: FGNode) => void;
  getNodeColor: (node: FGNode) => string;
  getLinkColor: (link: FGLink) => string;
  getLinkWidth: (link: FGLink) => number;
}) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || width < 20 || height < 20) {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(52, width / height, 1, 5000);
    camera.position.set(0, 0, 820);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    mount.replaceChildren(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const nodeMap = new Map(data.nodes.map((node) => [node.id, node]));
    const nodeMeshes: THREE.Mesh[] = [];

    for (const link of data.links) {
      const sourceId = typeof link.source === "object" ? (link.source as FGNode).id : link.source;
      const targetId = typeof link.target === "object" ? (link.target as FGNode).id : link.target;
      const source = nodeMap.get(String(sourceId));
      const target = nodeMap.get(String(targetId));
      if (!source || !target) {
        continue;
      }
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(source.x ?? 0, source.y ?? 0, source.z ?? 0),
        new THREE.Vector3(target.x ?? 0, target.y ?? 0, target.z ?? 0),
      ]);
      const material = new THREE.LineBasicMaterial({
        color: new THREE.Color(getLinkColor(link)),
        transparent: true,
        opacity: activeLinkNodeIds.has(source.id) && activeLinkNodeIds.has(target.id) ? 0.72 : 0.28,
        linewidth: Math.max(1, getLinkWidth(link)),
      });
      group.add(new THREE.Line(geometry, material));
    }

    for (const node of data.nodes) {
      const radius = Math.max(5, (NODE_RADIUS[node.kind] ?? 5) * 1.6);
      const geometry = new THREE.SphereGeometry(radius, 24, 16);
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(getNodeColor(node)),
        transparent: true,
        opacity: node.id === activeNodeId || activeLinkNodeIds.has(node.id) ? 0.98 : 0.72,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(node.x ?? 0, node.y ?? 0, node.z ?? 0);
      mesh.userData.node = node;
      nodeMeshes.push(mesh);
      group.add(mesh);
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const dragStart = { x: 0, y: 0 };
    let dragging = false;
    let moved = false;

    function setPointer(event: PointerEvent) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }

    function onPointerDown(event: PointerEvent) {
      dragging = true;
      moved = false;
      dragStart.x = event.clientX;
      dragStart.y = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    }

    function onPointerMove(event: PointerEvent) {
      if (!dragging) {
        return;
      }
      const dx = event.clientX - dragStart.x;
      const dy = event.clientY - dragStart.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) {
        moved = true;
      }
      group.rotation.y += dx * 0.006;
      group.rotation.x += dy * 0.004;
      group.rotation.x = clamp(group.rotation.x, -1.25, 1.25);
      dragStart.x = event.clientX;
      dragStart.y = event.clientY;
    }

    function onPointerUp(event: PointerEvent) {
      dragging = false;
      renderer.domElement.releasePointerCapture(event.pointerId);
      if (moved) {
        return;
      }
      setPointer(event);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(nodeMeshes, false)[0];
      const node = hit?.object.userData.node as FGNode | undefined;
      if (node) {
        onNodeClick(node);
      }
    }

    function onWheel(event: WheelEvent) {
      event.preventDefault();
      camera.position.z = clamp(camera.position.z + event.deltaY * 0.55, 180, 1600);
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    let frame = 0;
    function render() {
      frame = requestAnimationFrame(render);
      group.rotation.y += 0.0008;
      renderer.render(scene, camera);
    }
    render();

    return () => {
      cancelAnimationFrame(frame);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("wheel", onWheel);
      group.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose());
          } else {
            material.dispose();
          }
        }
      });
      renderer.dispose();
      mount.replaceChildren();
    };
  }, [activeLinkNodeIds, activeNodeId, data, getLinkColor, getLinkWidth, getNodeColor, height, onNodeClick, width]);

  return <div className="graphify__three-scene" ref={mountRef} />;
}

// ─── Node size mapping ────────────────────────────────────────────────────────
const NODE_RADIUS: Record<KnowledgeGraphNodeKind, number> = {
  corpus: 18,
  folder: 11,
  category: 9,
  document: 5,
  state: 7,
};

// ─── Link colour mapping ──────────────────────────────────────────────────────
const LINK_COLOR: Record<KnowledgeGraphLink["tone"], string> = {
  primary: "rgba(132,201,165,0.28)",
  category: "rgba(224,170,88,0.32)",
  state: "rgba(240,142,119,0.28)",
};
const LINK_COLOR_ACTIVE: Record<KnowledgeGraphLink["tone"], string> = {
  primary: "rgba(132,201,165,0.80)",
  category: "rgba(224,170,88,0.85)",
  state: "rgba(240,142,119,0.80)",
};

function GraphifyKnowledgeGraph({
  graph,
  activeCorpus,
  documents,
  selectedDocumentId,
  onSelectDocument,
  onOpenFolder,
  onOpenLibrary,
}: {
  graph: KnowledgeGraph;
  activeCorpus: CorpusRecord | null;
  documents: DocumentRecord[];
  selectedDocumentId: string | null;
  onSelectDocument: (id: string) => void;
  onOpenFolder: (id: string) => void;
  onOpenLibrary: () => void;
}) {
  const fg2dRef = useRef<ForceGraph2DMethods<FGNode, FGLink>>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 520 });
  const [focusedNodeId, setFocusedNodeId] = useState("corpus:active");
  const [query, setQuery] = useState("");
  const [focusNode, setFocusNode] = useState<KnowledgeGraphNode | null>(null);
  const [use3DGraph, setUse3DGraph] = useState(false);

  // Observe container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setDims({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setDims({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const selectedGraphNode = graph.nodes.find((n) => n.documentId === selectedDocumentId);
  const activeNodeId = selectedGraphNode?.id ?? focusedNodeId;

  const activeLinkNodeIds = useMemo(() => {
    const s = new Set<string>();
    graph.links
      .filter((l) => l.from === activeNodeId || l.to === activeNodeId)
      .forEach((l) => { s.add(l.from); s.add(l.to); });
    s.add(activeNodeId);
    return s;
  }, [graph.links, activeNodeId]);

  // Build filtered ForceGraph data
  const fgData = useMemo<FGData>(() => {
    const q = query.trim().toLowerCase();
    const nodes: FGNode[] = graph.nodes
      .filter((n) => !q || `${n.label} ${n.detail} ${n.kind}`.toLowerCase().includes(q))
      .map((n) => ({
        ...n,
        x: (n.x - 50) * 8,
        y: (n.y - 50) * 8,
        z: (hashToUnit(n.id) - 0.5) * 180,
      }));
    const ids = new Set(nodes.map((n) => n.id));
    const links: FGLink[] = graph.links
      .filter((l) => ids.has(l.from) && ids.has(l.to))
      .map((l) => ({ id: l.id, source: l.from, target: l.to, tone: l.tone }));
    return { nodes, links };
  }, [graph, query]);

  // Update focused node label panel
  useEffect(() => {
    const n = graph.nodes.find((node) => node.id === activeNodeId);
    setFocusNode(n ?? graph.nodes[0] ?? null);
  }, [activeNodeId, graph.nodes]);

  // Zoom-to-fit after data changes
  useEffect(() => {
    const t = setTimeout(() => {
      if (!use3DGraph) {
        fg2dRef.current?.zoomToFit?.(500, 56);
      }
    }, 700);
    return () => clearTimeout(t);
  }, [fgData, use3DGraph]);

  const handleNodeClick = useCallback((node: FGNode) => {
    setFocusedNodeId(node.id);
    setFocusNode(node);
    if (node.kind === "corpus") {
      onOpenLibrary();
    }
    if (node.folderId) {
      onOpenFolder(node.folderId);
    }
    if (node.documentId) {
      onSelectDocument(node.documentId);
    }

    if (!use3DGraph) {
      fg2dRef.current?.centerAt(node.x ?? 0, node.y ?? 0, 450);
      fg2dRef.current?.zoom(2.2, 450);
    }
  }, [onOpenLibrary, onOpenFolder, onSelectDocument, use3DGraph]);

  const getLinkColor = useCallback((link: FGLink) => {
    const src = typeof link.source === "object" ? (link.source as FGNode).id : link.source;
    const tgt = typeof link.target === "object" ? (link.target as FGNode).id : link.target;
    const isActive = activeLinkNodeIds.has(src) && activeLinkNodeIds.has(tgt);
    return isActive ? LINK_COLOR_ACTIVE[link.tone] : LINK_COLOR[link.tone];
  }, [activeLinkNodeIds]);

  const getLinkWidth = useCallback((link: FGLink) => {
    const src = typeof link.source === "object" ? (link.source as FGNode).id : link.source;
    const tgt = typeof link.target === "object" ? (link.target as FGNode).id : link.target;
    return activeLinkNodeIds.has(src) && activeLinkNodeIds.has(tgt) ? 1.2 : 0.4;
  }, [activeLinkNodeIds]);

  const getNodeColor = useCallback((node: FGNode) => {
    const isActive = node.id === activeNodeId;
    const isRelated = activeLinkNodeIds.has(node.id);
    if (!query || isActive || isRelated) {
      return node.color;
    }
    return "rgba(150,150,150,0.45)";
  }, [activeLinkNodeIds, activeNodeId, query]);

  const paintNode = useCallback((node: FGNode, ctx: CanvasRenderingContext2D) => {
    const r = NODE_RADIUS[node.kind] ?? 5;
    const isActive = node.id === activeNodeId;
    const isRelated = activeLinkNodeIds.has(node.id);
    const alpha = !query && !isActive && !isRelated ? 0.55 : 1;

    ctx.globalAlpha = alpha;
    ctx.shadowColor = node.color;
    ctx.shadowBlur = isActive ? 18 : node.kind === "corpus" || node.kind === "category" ? 8 : 0;
    ctx.beginPath();
    ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.fill();

    if (isActive) {
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.shadowBlur = 0;
    if (node.kind === "corpus" || node.kind === "category" || node.kind === "folder" || isActive) {
      const fs = node.kind === "corpus" ? 5 : node.kind === "category" ? 3.8 : 3.2;
      ctx.font = `${node.kind === "corpus" ? 700 : 500} ${fs}px "Segoe UI", sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(20,24,22,0.9)";
      ctx.fillText(node.label.slice(0, 22), node.x ?? 0, (node.y ?? 0) + r + fs * 0.9);
    }

    ctx.globalAlpha = 1;
  }, [activeLinkNodeIds, activeNodeId, query]);

  const paintNodePointerArea = useCallback((node: FGNode, color: string, ctx: CanvasRenderingContext2D) => {
    const r = Math.max(10, (NODE_RADIUS[node.kind] ?? 5) + 6);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
    ctx.fill();
  }, []);

  function resetView() {
    setFocusedNodeId("corpus:active");
    setQuery("");
    if (!use3DGraph) {
      fg2dRef.current?.zoomToFit?.(500, 56);
    }
  }

  return (
    <div className="graphify">
      <div className="graphify__toolbar">
        <label className="graphify__search">
          <span>Search graph</span>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Policy, folder, warning..."
          />
        </label>
        <div className="graphify__controls" aria-label="Knowledge graph controls">
          <button type="button" className="ghost-button button--small" onClick={() => fg2dRef.current?.zoom((fg2dRef.current?.zoom() ?? 1) * 0.75, 200)} disabled={use3DGraph}>
            Zoom −
          </button>
          <button type="button" className="ghost-button button--small" onClick={() => fg2dRef.current?.zoom((fg2dRef.current?.zoom() ?? 1) * 1.33, 200)} disabled={use3DGraph}>
            Zoom +
          </button>
          <button type="button" className="ghost-button button--small" onClick={resetView}>
            Reset
          </button>
          <button type="button" className="ghost-button button--small" onClick={() => setUse3DGraph((current) => !current)}>
            {use3DGraph ? "Use 2D" : "Use 3D"}
          </button>
        </div>
      </div>

      <div className="graphify__stage" ref={containerRef} role="application" aria-label="Graphify knowledge graph explorer">
        {use3DGraph ? (
          <ThreeKnowledgeScene
            data={fgData}
            width={dims.w}
            height={dims.h}
            activeNodeId={activeNodeId}
            activeLinkNodeIds={activeLinkNodeIds}
            onNodeClick={handleNodeClick}
            getNodeColor={getNodeColor}
            getLinkColor={getLinkColor}
            getLinkWidth={getLinkWidth}
          />
        ) : (
          <ForceGraph2D<FGNode, FGLink>
            ref={fg2dRef}
            width={dims.w}
            height={dims.h}
            graphData={fgData}
            backgroundColor="transparent"
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => "replace"}
            nodePointerAreaPaint={paintNodePointerArea}
            nodeRelSize={1}
            nodeVal={(n) => (NODE_RADIUS[n.kind] ?? 5) ** 2}
            linkColor={getLinkColor}
            linkWidth={getLinkWidth}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={getLinkWidth}
            linkDirectionalParticleColor={getLinkColor}
            linkDirectionalParticleSpeed={0.003}
            onNodeClick={handleNodeClick}
            nodeLabel={(n) => `${n.label}: ${n.detail}`}
            cooldownTicks={160}
            warmupTicks={60}
            d3AlphaDecay={0.018}
            d3VelocityDecay={0.28}
            enableNodeDrag
            enableZoomInteraction
            enablePanInteraction
          />
        )}
      </div>

      <div className="graphify__insight">
        <div>
          <span className="panel__eyebrow">Focused node</span>
          <strong>{focusNode?.label ?? activeCorpus?.name ?? "Knowledge graph"}</strong>
          <p>{focusNode?.detail ?? "Select a node to inspect its relationships."}</p>
        </div>
        <div className="graphify__stats">
          <span>{graph.nodes.length} nodes</span>
          <span>{graph.links.length} links</span>
          <span>{documents.length} files</span>
        </div>
      </div>
    </div>
  );
}

export function LibraryPage() {
  const [corpora, setCorpora] = useState<CorpusRecord[]>([]);
  const [documentsState, setDocumentsState] = useState<DocumentListResponse | null>(null);
  const [health, setHealth] = useState<CorpusHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newCorpusName, setNewCorpusName] = useState("HR Resources");
  const [newCorpusPath, setNewCorpusPath] = useState("data/raw_pdfs/HR Resources");
  const [replacementPath, setReplacementPath] = useState("data/raw_pdfs/HR Resources");
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [showCorpusSwitcher, setShowCorpusSwitcher] = useState(false);
  const [libraryDrawerOpen, setLibraryDrawerOpen] = useState(false);
  const [expandedFolderIds, setExpandedFolderIds] = useState<string[]>(["root"]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

  const activeCorpus = corpora.find((item) => item.active) ?? corpora[0] ?? null;
  const documents = documentsState?.documents ?? [];
  const activeJobs = activeCorpus?.active_jobs ?? documentsState?.active_jobs ?? [];

  async function loadLibrary() {
    setLoading(true);
    setError(null);
    try {
      const [corporaPayload, documentsPayload] = await Promise.all([api.listCorpora(), api.listDocuments()]);
      setCorpora(corporaPayload.corpora);
      setDocumentsState(documentsPayload);
      const active = corporaPayload.corpora.find((item) => item.active) ?? corporaPayload.corpora[0];
      if (active) {
        setHealth(await api.getCorpusHealth(active.id));
        setReplacementPath(active.source_path);
      } else {
        setHealth(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load the HR library.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadLibrary();
  }, []);

  async function runAction(label: string, task: () => Promise<unknown>) {
    setWorking(true);
    setError(null);
    try {
      await task();
      await loadLibrary();
    } catch (err) {
      const message = err instanceof Error ? err.message : `${label} failed.`;
      setError(message);
    } finally {
      setWorking(false);
    }
  }

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await runAction("Upload document", () => api.uploadDocument(file));
    event.target.value = "";
  }

  async function handleDelete(docName: string) {
    setDeletingDoc(docName);
    try {
      await runAction("Delete document", () => api.deleteDocument(docName));
    } finally {
      setDeletingDoc(null);
    }
  }

  const explorerTree = useMemo(() => buildExplorerTree(documents), [documents]);
  const knowledgeGraph = useMemo(
    () => buildKnowledgeGraph({ activeCorpus, documents, explorerTree }),
    [activeCorpus, documents, explorerTree],
  );
  const selectedFile = useMemo(() => findSelectedDocument(explorerTree, selectedDocumentId), [explorerTree, selectedDocumentId]);
  const selectedDocument = selectedFile?.document ?? null;
  const selectedBreadcrumbs = useMemo(
    () => selectedFile?.relativePath.split(/[\\/]/).filter(Boolean) ?? [],
    [selectedFile],
  );

  useEffect(() => {
    const allFolderIds = new Set(collectFolderIds(explorerTree));
    setExpandedFolderIds((current) => current.filter((id) => allFolderIds.has(id)));
    if (!selectedDocumentId) {
      const firstFile = findFirstFile(explorerTree);
      setSelectedDocumentId(firstFile?.id ?? null);
    }
  }, [explorerTree, selectedDocumentId]);

  useEffect(() => {
    if (selectedFile?.document?.source_relative_path) {
      const parts = selectedFile.document.source_relative_path.split(/[\\/]/).filter(Boolean);
      const folderLineage = parts
        .slice(0, -1)
        .map((_, index, array) => `folder:${array.slice(0, index + 1).join("/")}`);
      setExpandedFolderIds((current) => [...new Set([...current, "root", ...folderLineage])]);
    }
  }, [selectedFile]);

  if (loading) {
    return <LibrarySkeleton />;
  }

  return (
    <section className="page-stack">
      <input
        id="library-upload-input"
        type="file"
        accept=".pdf,.docx,.xlsx,.csv"
        style={{ display: "none" }}
        onChange={(event) => void handleUpload(event)}
      />

      <div className="section-heading">
        <div>
          <span className="panel__eyebrow">HR knowledge library</span>
          <h2>Manage HR documents, corpus status, and indexing.</h2>
          <p>Keep policy files organized, review corpus health, and switch between HR knowledge sets.</p>
        </div>
        <div className="section-heading__actions section-heading__actions--popover">
          <button
            type="button"
            className="upload-button"
            onClick={() => setShowCorpusSwitcher((current) => !current)}
          >
            Change corpus
          </button>
          {showCorpusSwitcher ? (
            <section className="panel section-heading__popover">
              <div className="panel__header">
                <div>
                  <span className="panel__eyebrow">Corpus switcher</span>
                  <h2>Load an indexed corpus</h2>
                  <p>Switch to another ready corpus and keep its own chunks, citations, and references.</p>
                </div>
              </div>
              <div className="panel-flow">
                {corpora.map((corpus) => (
                  <div key={corpus.id} className="document-card document-card--registry">
                    <div className="document-card__top">
                      <div className="document-card__title">
                        <span className="panel__eyebrow">{corpus.active ? "Active" : "Available"}</span>
                        <h3>{corpus.name}</h3>
                        <p className="document-card__quiet">{corpus.source_path}</p>
                      </div>
                      {!corpus.active ? (
                        <button
                          className="ghost-button button--small"
                          onClick={() =>
                            void runAction("Activate corpus", async () => {
                              await api.activateCorpus(corpus.id);
                              setShowCorpusSwitcher(false);
                            })
                          }
                        >
                          Load corpus
                        </button>
                      ) : (
                        <span className="status-chip status-chip--success">Current</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
        </div>
      </div>

      <div className="library-layout library-layout--workspace">
        <aside className="library-sidebar">
          <section className="panel">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Active corpus</span>
                <h2>{activeCorpus?.name ?? "No active corpus"}</h2>
                <p>{activeCorpus?.source_path ?? "No source path set."}</p>
              </div>
              {activeCorpus ? (
                <RadialActionMenu
                  label="Corpus actions"
                  actions={[
                    {
                      id: "rebuild",
                      label: "Rebuild",
                      icon: RefreshIcon,
                      angle: -78,
                      onClick: () => void runAction("Rebuild corpus", () => api.rebuildCorpus(activeCorpus.id)),
                    },
                    {
                      id: "reset",
                      label: "Reset",
                      icon: ResetIcon,
                      angle: -38,
                      onClick: () => void runAction("Reset corpus", () => api.resetCorpus(activeCorpus.id)),
                    },
                    {
                      id: "replace-source",
                      label: "Replace source",
                      icon: FolderIcon,
                      angle: 2,
                      onClick: () =>
                        void runAction("Replace source", () =>
                          api.replaceCorpusSource(activeCorpus.id, replacementPath.trim()),
                        ),
                    },
                    {
                      id: "upload",
                      label: "Upload document",
                      icon: UploadIcon,
                      angle: 42,
                      onClick: () => document.getElementById("library-upload-input")?.click(),
                    },
                  ]}
                />
              ) : null}
            </div>

            <section className="summary-grid summary-grid--four">
              <div className="metric-card">
                <span>Source files</span>
                <strong>{health?.source_file_count ?? activeCorpus?.source_file_count ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Indexed docs</span>
                <strong>{activeCorpus?.document_count ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Chunks</span>
                <strong>{activeCorpus?.chunk_count ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Drift</span>
                <strong>{health?.drift_state ?? activeCorpus?.drift_state ?? "unknown"}</strong>
              </div>
            </section>

            {health?.drift_state && health.drift_state !== "current" ? (
              <div className="error-banner">
                Corpus is {health.drift_state}: {health.new_source_count} new, {health.missing_source_count} missing, {health.changed_source_count} changed.
              </div>
            ) : null}
            {activeCorpus?.last_error ? <div className="error-banner">Last rebuild failed: {activeCorpus.last_error}</div> : null}
          </section>

          <section className="panel">
            <div className="panel__header">
              <div>
                <span className="panel__eyebrow">Ingestion management</span>
                <h2>Corpus source control</h2>
              </div>
            </div>
            <div className="panel-flow">
              <label className="field-stack">
                <span className="field-label">New corpus name</span>
                <input type="text" value={newCorpusName} onChange={(event) => setNewCorpusName(event.target.value)} />
              </label>
              <label className="field-stack">
                <span className="field-label">New corpus source path</span>
                <input type="text" value={newCorpusPath} onChange={(event) => setNewCorpusPath(event.target.value)} />
              </label>
              <button
                type="button"
                onClick={() =>
                  void runAction("Create corpus", () =>
                    api.createCorpus({ name: newCorpusName.trim(), source_path: newCorpusPath.trim() }),
                  )
                }
                disabled={!newCorpusName.trim() || !newCorpusPath.trim() || working}
              >
                Create corpus
              </button>
              <label className="field-stack">
                <span className="field-label">Replace active source path</span>
                <input type="text" value={replacementPath} onChange={(event) => setReplacementPath(event.target.value)} />
              </label>
              <button
                type="button"
                className="ghost-button"
                onClick={() =>
                  activeCorpus
                    ? void runAction("Replace source", () => api.replaceCorpusSource(activeCorpus.id, replacementPath.trim()))
                    : undefined
                }
                disabled={!activeCorpus || !replacementPath.trim() || working}
              >
                Replace source
              </button>
            </div>
          </section>
        </aside>

        <section className="library-workspace">
          <section className="library-canvas library-canvas--graph-space">
            <div className="graph-space__hud">
              <div>
                <span className="panel__eyebrow">Graphify explorer</span>
                <h2>Knowledge graph</h2>
                <p>Traverse corpus structure, policy categories, document states, and source files from one explorable map.</p>
              </div>
            </div>

            <GraphifyKnowledgeGraph
              graph={knowledgeGraph}
              activeCorpus={activeCorpus}
              documents={documents}
              selectedDocumentId={selectedDocumentId}
              onOpenLibrary={() => setLibraryDrawerOpen(true)}
              onOpenFolder={(id) => {
                setExpandedFolderIds((current) => [...new Set([...current, "root", id])]);
                setLibraryDrawerOpen(true);
              }}
              onSelectDocument={(id) => {
                setSelectedDocumentId(id);
                const target = findSelectedDocument(explorerTree, id);
                if (target?.relativePath) {
                  const parts = target.relativePath.split(/[\\/]/).filter(Boolean);
                  const folderLineage = parts
                    .slice(0, -1)
                    .map((_, index, array) => `folder:${array.slice(0, index + 1).join("/")}`);
                  setExpandedFolderIds((current) => [...new Set([...current, "root", ...folderLineage])]);
                }
                setLibraryDrawerOpen(true);
              }}
            />

            <ProgressCard
              active={working || activeJobs.length > 0}
              title={working ? "Updating the HR library" : "Synchronizing corpus activity"}
              statuses={
                working
                  ? ["Validating request...", "Refreshing corpus state...", "Reloading the library..."]
                  : ["Refreshing the registry and active job feed...", "Reconciling corpus jobs...", "Updating the library canvas..."]
              }
              stalledLabel="The pipeline is still working. You can refresh the library or inspect the latest corpus state."
              actionLabel="Refresh now"
              onAction={() => void loadLibrary()}
            />

            {error ? <div className="error-banner">{error}</div> : null}
          </section>

          <button
            type="button"
            className={`library-drawer__tab ${libraryDrawerOpen ? "library-drawer__tab--open" : ""}`}
            onClick={() => setLibraryDrawerOpen((current) => !current)}
            aria-expanded={libraryDrawerOpen}
            aria-controls="hr-library-drawer"
          >
            <LibraryIcon width={16} height={16} />
            <span>Library</span>
          </button>

          <aside id="hr-library-drawer" className={`library-drawer ${libraryDrawerOpen ? "library-drawer--open" : ""}`}>
            <div className="library-drawer__glass">
              <div className="library-drawer__header">
                <div>
                  <span className="panel__eyebrow">HR document library</span>
                  <h2>Workspace explorer</h2>
                  <p>Navigate folders and files the way you would in an IDE workspace.</p>
                </div>
                <button
                  type="button"
                  className="ghost-button button--icon"
                  onClick={() => setLibraryDrawerOpen(false)}
                  aria-label="Close library drawer"
                >
                  <CloseIcon width={16} height={16} />
                </button>
              </div>

              <div className="library-drawer__body">
                <aside className="library-drawer__tree">
                  <div className="library-drawer__section-label">Workspace</div>
                  {renderTree({
                    node: explorerTree,
                    depth: 0,
                    expandedFolderIds: new Set(expandedFolderIds),
                    onToggleFolder: (id) =>
                      setExpandedFolderIds((current) =>
                        current.includes(id) ? current.filter((entry) => entry !== id) : [...current, id],
                      ),
                    selectedDocumentId,
                    onSelectDocument: (id) => {
                      setSelectedDocumentId(id);
                      const target = findSelectedDocument(explorerTree, id);
                      if (target?.relativePath) {
                        const parts = target.relativePath.split(/[\\/]/).filter(Boolean);
                        const folderLineage = parts
                          .slice(0, -1)
                          .map((_, index, array) => `folder:${array.slice(0, index + 1).join("/")}`);
                        setExpandedFolderIds((current) => [...new Set([...current, "root", ...folderLineage])]);
                      }
                    },
                  })}
                </aside>

                <section className="library-drawer__preview">
                  {selectedDocument ? (
                    <>
                      <div className="library-drawer__preview-header">
                        <div>
                          <span className="panel__eyebrow">Selected file</span>
                          <div className="library-drawer__breadcrumbs" aria-label="Selected file path">
                            {selectedBreadcrumbs.map((segment, index) => (
                              <span key={`${segment}-${index}`} className="library-drawer__breadcrumb">
                                <span>{segment}</span>
                                {index < selectedBreadcrumbs.length - 1 ? (
                                  <ChevronIcon width={11} height={11} />
                                ) : null}
                              </span>
                            ))}
                          </div>
                          <h3>{selectedDocument.doc_name}</h3>
                          <p>{selectedDocument.doc_summary ?? "Open extracted details, metadata, and warnings for this HR document."}</p>
                        </div>
                      </div>
                      <div className="library-drawer__preview-scroll">
                        <section className="library-drawer__inspector">
                          <div className="library-drawer__inspector-grid">
                            <div className="library-drawer__inspector-item">
                              <span>Format</span>
                              <strong>{selectedDocument.doc_format.toUpperCase()}</strong>
                            </div>
                            <div className="library-drawer__inspector-item">
                              <span>Pages</span>
                              <strong>{selectedDocument.page_count || "Unknown"}</strong>
                            </div>
                            <div className="library-drawer__inspector-item">
                              <span>Tables</span>
                              <strong>{selectedDocument.table_count}</strong>
                            </div>
                            <div className="library-drawer__inspector-item">
                              <span>Category</span>
                              <strong>{selectedDocument.policy_category ?? "Unclassified"}</strong>
                            </div>
                            <div className="library-drawer__inspector-item">
                              <span>Source level</span>
                              <strong>{selectedDocument.source_level ?? "Unknown"}</strong>
                            </div>
                            <div className="library-drawer__inspector-item">
                              <span>Effective date</span>
                              <strong>{formatDocDate(selectedDocument.effective_date ?? selectedDocument.doc_date) ?? "Unknown"}</strong>
                            </div>
                          </div>

                          <div className="library-drawer__inspector-section">
                            <span className="panel__eyebrow">Document notes</span>
                            <p>
                              {selectedDocument.doc_summary ??
                                "No extracted summary is available for this document yet."}
                            </p>
                          </div>

                          <div className="library-drawer__inspector-section">
                            <span className="panel__eyebrow">Extraction warnings</span>
                            {selectedDocument.warnings.length ? (
                              <ul className="library-drawer__warning-list">
                                {selectedDocument.warnings.map((warning) => (
                                  <li key={warning}>{warning}</li>
                                ))}
                              </ul>
                            ) : (
                              <p>No extraction warnings for this file.</p>
                            )}
                          </div>

                          <div className="library-drawer__inspector-actions">
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={() => void handleDelete(selectedDocument.doc_name)}
                              disabled={deletingDoc === selectedDocument.doc_name}
                            >
                              {deletingDoc === selectedDocument.doc_name ? "Removing..." : "Remove file"}
                            </button>
                          </div>
                        </section>
                      </div>
                    </>
                  ) : (
                    <div className="file-explorer__empty">Choose a document from the workspace tree to inspect its extracted details.</div>
                  )}
                </section>
              </div>
            </div>
          </aside>
        </section>
      </div>
    </section>
  );
}
