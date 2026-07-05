import { useQuery } from "@tanstack/react-query"
import { getCollections } from "@/lib/api/client"

export function useCollections() {
  return useQuery({
    queryKey: ["collections"],
    queryFn: getCollections,
    staleTime: 30_000,
  })
}
