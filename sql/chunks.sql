create table chunks (
  chunk_id text primary key,
  document_id text not null references documents(document_id),
  user_id text not null,
  org_id text not null,
  text text not null,
  metadata jsonb default '{}',
  embedding vector(1536),
  created_at timestamptz not null default now()
);